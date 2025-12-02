import aiohttp
import json
import asyncio, os
import astrbot.api.message_components as Comp
from astrbot.api.message_components import File
from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, register
from astrbot.api import logger, AstrBotConfig
from astrbot.core.utils.session_waiter import session_waiter, SessionController

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import colorsys
import re
from io import BytesIO
import pathlib

# ========== API URLs ==========
MOJANG_API_URL = "https://api.mojang.com/users/profiles/minecraft/{username}"
STARLIGHT_RENDER_URL = "https://starlightskins.lunareclipse.studio/render/{rendertype}/{uuid}/{rendercrop}"
WALLPAPER_API_URL = "https://starlightskins.lunareclipse.studio/render/wallpaper/{wallpaper_id}/{playernames}"

# ========== 渲染类型配置 ==========
VALID_RENDERTYPES = {
    "default", "marching", "walking", "crouching", "crossed", "criss_cross",
    "ultimate", "isometric", "head", "custom", "cheering", "relaxing",
    "trudging", "cowering", "pointing", "lunging", "dungeons", "facepalm",
    "sleeping", "dead", "archer", "kicking", "mojavatar", "reading",
    "high_ground", "clown", "bitzel", "pixel", "ornament", "skin", "profile"
}

WALLPAPER_CONFIGS = {
    "herobrine_hill": 1,
    "quick_hide": 3,
    "malevolent": 1,
    "off_to_the_stars": 1,
    "wheat": 1
}

# ========== 默认配置 ==========
DEFAULT_RENDERTYPE = "default"
DEFAULT_RENDERCROP = "full"
SKIN_RENDERCROP = "default"
DEFAULT_WALLPAPER = "herobrine_hill"

# ========== 自定义皮肤配置 ==========
FILE_WAIT_TIMEOUT = 30
DEFAULT_CAMERA_POSITION = {"x": 0, "y": 0, "z": 0}
DEFAULT_CAMERA_FOCAL_POINT = {"x": 0, "y": 0, "z": 0}
CAMERA_PRESETS = {}
FOCAL_PRESETS = {}

# ========== Passport 配置 ==========
PASSPORT_CONFIG = {
    'output_size': (1920, 3840),
    'text_config': {
        'uid': {
            'center': (0.985, 0.685),
            'align': 'R',
            'size': 220,
            'max_width': 10000,
            'fonts': ['AdobeGothicStd-Bold.otf'],
            'opacity': 255,
            'shadow': 3
        },
        'title': {
            'center': (0.985, 0.725),
            'align': 'R',
            'size': 80,
            'max_width': 100000000,
            'fonts': ['SourceHanSansCN-Normal.ttf'],
            'opacity': 230,
            'shadow': 2
        },
        'message': {
            'center': (0.5, 0.82),
            'align': 'M',
            'size': 70,
            'max_width': 100000000,
            'fonts': ['字魂白润体(商用需授权).ttf'],
            'opacity': 200,
            'shadow': 0
        },
        'wish': {
            'center': (0.16, 0.945),
            'align': 'L',
            'size': 60,
            'max_width': 5000000000,
            'fonts': ['字魂太阿楷书(商用需授权).ttf'],
            'opacity': 180,
            'shadow': 0
        },
        'watermark': {
            'center': (0.15, 0.965),
            'align': 'L',
            'size': 60,
            'max_width': 10000,
            'fonts': ['ROCKB.TTF'],
            'opacity': 180,
            'shadow': 0
        }
    },
    'watermark_text': "Let's play a lifelong club",
    'uid_max_length': 1635
}

# ========== 工具函数 ==========
def _srgb_to_linear(x):
    """sRGB to linear RGB"""
    a = 0.055
    return np.where(x <= 0.04045, x / 12.92, ((x + a) / (1 + a)) ** 2.4)

def _linear_to_srgb(x):
    """linear RGB to sRGB"""
    a = 0.055
    return np.where(x <= 0.0031308, 12.92 * x, (1 + a) * (np.clip(x,0,1) ** (1/2.4)) - a)

def _hsl_to_rgb_vec(h, s, l):
    """Vectorized HSL to RGB conversion"""
    c = (1 - np.abs(2 * l - 1)) * s
    hp = (h * 6.0) % 6.0
    x = c * (1 - np.abs((hp % 2) - 1))

    z = np.zeros_like(h)
    r1 = np.where((0 <= hp) & (hp < 1), c,
         np.where((1 <= hp) & (hp < 2), x,
         np.where((2 <= hp) & (hp < 3), z,
         np.where((3 <= hp) & (hp < 4), z,
         np.where((4 <= hp) & (hp < 5), x,
         np.where((5 <= hp) & (hp < 6), c, z))))))
    g1 = np.where((0 <= hp) & (hp < 1), x,
         np.where((1 <= hp) & (hp < 2), c,
         np.where((2 <= hp) & (hp < 3), c,
         np.where((3 <= hp) & (hp < 4), x,
         np.where((4 <= hp) & (hp < 5), z,
         np.where((5 <= hp) & (hp < 6), z, z))))))
    b1 = np.where((0 <= hp) & (hp < 1), z,
         np.where((1 <= hp) & (hp < 2), z,
         np.where((2 <= hp) & (hp < 3), x,
         np.where((3 <= hp) & (hp < 4), c,
         np.where((4 <= hp) & (hp < 5), c,
         np.where((5 <= hp) & (hp < 6), x, z))))))

    m = l - c / 2.0
    r = r1 + m
    g = g1 + m
    b = b1 + m
    rgb = np.stack([r, g, b], axis=-1)
    return np.clip(rgb, 0, 1)

def gray_to_hsv_tint(arr, theme_rgb):
    """Apply color tint to grayscale image"""
    gray = arr[..., 0].astype(np.float32) / 255.0
    alpha = arr[..., 3]
    r, g, b = [c / 255.0 for c in theme_rgb]
    h, l_, s = colorsys.rgb_to_hls(r, g, b)
    H = np.full_like(gray, h, dtype=np.float32)
    S = np.full_like(gray, s, dtype=np.float32)
    L = gray
    rgb = _hsl_to_rgb_vec(H, S, L)
    rgb_u8 = (rgb * 255.0 + 0.5).astype(np.uint8)
    return np.dstack((rgb_u8, alpha))

def parse_theme_rgb(text):
    """Parse RGB string to tuple"""
    if text is None:
        raise ValueError("RGB is empty")
    s = str(text)
    if ',' in s:
        parts = s.split(',')
    elif '，' in s:
        parts = s.split('，')
    else:
        raise ValueError('Invalid RGB format')
    vals = [int(num.strip()) for num in parts]
    if len(vals) != 3:
        raise ValueError('RGB must have 3 numbers')
    return tuple(vals)

def wrap_text_by_width(draw, font, text, max_width):
    """Wrap text by pixel width"""
    text = "" if text is None else str(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    lines = []
    for raw in text.split("\n"):
        current = ""
        for ch in raw:
            trial = current + ch
            w = draw.textlength(trial, font=font)
            if w <= max_width or not current:
                current = trial
            else:
                lines.append(current.rstrip())
                current = ch
        if current:
            lines.append(current.rstrip())
        if raw == "" and (not lines or lines[-1] != ""):
            lines.append("")
    return lines

def draw_centered_text(draw, font, text, cx, cy, opacity):
    """Draw centered text"""
    bbox = draw.textbbox((0,0), text, font=font)
    w = bbox[2] - bbox[0]; h = bbox[3] - bbox[1]
    x = cx - w//2; y = cy - h//2
    draw.text((x, y), text, font=font, fill=(255,255,255,opacity))

def draw_text_with_wrap_and_left_bottom_align(draw, font, text, x_left, y_bottom, max_width, opacity, line_spacing=1.0):
    """Draw left-aligned multi-line text"""
    if not text:
        return
    lines = wrap_text_by_width(draw, font, text, max_width)
    if not lines:
        return
    heights = []
    for ln in lines:
        bbox = draw.textbbox((0,0), ln, font=font)
        heights.append(bbox[3] - bbox[1])
    line_h = max(heights) if heights else font.size
    total_h = int(line_h * (len(lines) + (len(lines)-1)*(line_spacing-1))) if lines else 0
    y_start = y_bottom - total_h
    y = y_start
    for ln in lines:
        draw.text((x_left, y), ln, font=font, fill=(255,255,255,opacity))
        y += int(line_h * line_spacing)

def validate_rendertype(rendertype: str) -> tuple[bool, str | None]:
    """验证渲染类型"""
    if rendertype not in VALID_RENDERTYPES:
        valid_types_sample = ", ".join(sorted(VALID_RENDERTYPES)[:5]) + "..."
        error_msg = (f"未知的渲染类型 '{rendertype}'。\n"
                    f"有效类型例如: {valid_types_sample}\n"
                    f"输入 /skinhelp 查看完整列表。")
        return False, error_msg
    return True, None

def validate_wallpaper(wallpaper_id: str) -> tuple[bool, str | None, int]:
    """验证壁纸ID"""
    if wallpaper_id not in WALLPAPER_CONFIGS:
        available_wallpapers = ", ".join(sorted(WALLPAPER_CONFIGS.keys()))
        error_msg = (f"未知的壁纸类型 '{wallpaper_id}'。\n"
                    f"可用壁纸: {available_wallpapers}\n"
                    f"输入 /skinhelp 查看详细信息。")
        return False, error_msg, 0
    return True, None, WALLPAPER_CONFIGS[wallpaper_id]

async def get_player_uuid(session, username: str) -> tuple[str | None, str | None]:
    """获取玩家UUID"""
    mojang_url = MOJANG_API_URL.format(username=username)
    logger.info(f"正在为 {username} 异步查询 UUID...")
    
    try:
        async with session.get(mojang_url) as response:
            if response.status != 200:
                logger.warning(f"Mojang API 玩家 {username} 未找到 (状态: {response.status})。")
                return None, f"错误：找不到玩家 '{username}'。"
            
            player_data = await response.json()
            uuid = player_data.get("id")
            
            if not uuid:
                logger.error(f"Mojang API 响应中未找到 {username} 的 UUID。")
                return None, "获取玩家数据时出错。"
            
            logger.info(f"成功获取 {username} 的 UUID: {uuid}")
            return uuid, None
            
    except aiohttp.ClientError as e:
        logger.error(f"为 {username} 获取 UUID 时发生 aiohttp ClientError: {e}")
        return None, "查询玩家信息时发生网络错误，请稍后再试。"
    except Exception as e:
        logger.error(f"获取 {username} 的 UUID 时发生未知错误: {e}", exc_info=True)
        return None, "查询玩家信息时发生内部错误。"

async def process_skin_command(session, username: str, rendertype: str):
    """处理皮肤命令的核心逻辑"""
    rendertype_lower = rendertype.lower()
    is_valid, error_msg = validate_rendertype(rendertype_lower)
    if not is_valid:
        return error_msg

    uuid, error_msg = await get_player_uuid(session, username)
    if error_msg:
        return error_msg

    rendercrop = SKIN_RENDERCROP if rendertype_lower == "skin" else DEFAULT_RENDERCROP
    render_url = STARLIGHT_RENDER_URL.format(
        rendertype=rendertype_lower,
        uuid=uuid,
        rendercrop=rendercrop
    )

    logger.info(f"为 {username} 生成渲染 URL: {render_url}")
    render_desc = f"'{rendertype_lower}' 渲染"
    
    return [
        Comp.Plain(f"这是 {username} 的 {render_desc}：\n"),
        Comp.Image.fromURL(url=render_url)
    ]

async def process_wallpaper_command(session, wallpaper_id: str, usernames: list[str]):
    """处理壁纸命令的核心逻辑"""
    wallpaper_lower = wallpaper_id.lower()
    is_valid, error_msg, max_players = validate_wallpaper(wallpaper_lower)
    if not is_valid:
        return error_msg

    if not usernames:
        return (
            f"错误：壁纸 '{wallpaper_lower}' 至少需要1个玩家名称。\n"
            f"用法：/wallpaper {wallpaper_lower} <玩家名1> [玩家名2] ...\n"
            f"该壁纸最多支持 {max_players} 个玩家。"
        )

    actual_usernames = usernames
    warning_msg = ""
    if len(actual_usernames) > max_players:
        warning_msg = f"⚠️ 注意：壁纸 '{wallpaper_lower}' 最多支持 {max_players} 个玩家，已自动截取前 {max_players} 个。\n\n"
        actual_usernames = actual_usernames[:max_players]

    player_uuids = []
    failed_players = []

    for username in actual_usernames:
        uuid, error_msg_uuid = await get_player_uuid(session, username)
        if error_msg_uuid:
            failed_players.append(username)
            logger.warning(f"无法获取玩家 {username} 的 UUID，跳过该玩家")
        else:
            player_uuids.append(uuid)

    if not player_uuids:
        error_list = "\n".join([f"• {player}" for player in failed_players])
        return (
            f"错误：无法获取任何玩家的 UUID。\n"
            f"失败的玩家：\n{error_list}"
        )

    if failed_players:
        failed_list = ", ".join(failed_players)
        warning_msg += f"⚠️ 以下玩家未找到，已跳过：{failed_list}\n\n"

    player_uuids_path = ",".join(player_uuids)
    wallpaper_url = WALLPAPER_API_URL.format(
        wallpaper_id=wallpaper_lower,
        playernames=player_uuids_path
    )

    logger.info(f"为壁纸 '{wallpaper_lower}' 生成 URL（{len(player_uuids)} 个玩家）: {wallpaper_url}")

    success_players = [name for name in actual_usernames if name not in failed_players]
    players_desc = ", ".join(success_players)
    
    return [
        Comp.Plain(f"{warning_msg}这是壁纸 '{wallpaper_lower}' (玩家: {players_desc})：\n"),
        Comp.Image.fromURL(url=wallpaper_url)
    ]

async def upload_and_render_custom_skin(session, uuid, file_url, username, camera_position=None, camera_focal_point=None):
    """上传并渲染自定义皮肤"""
    # 这里应该是自定义皮肤渲染的逻辑
    # 由于原代码中没有完整实现，这里返回一个占位消息
    return [
        Comp.Plain(f"自定义皮肤渲染功能 - 玩家: {username}\n"),
        Comp.Plain(f"模型文件: {file_url}\n"),
        Comp.Plain("自定义皮肤渲染功能正在开发中...")
    ]

def replace_hyphens_with_spaces(text: str) -> str:
    """将文本中的^替换为空格，允许中文输入"""
    if text is None:
        return ""
    return str(text).replace('^', ' ')

def get_help_text():
    """获取帮助文本"""
    return (
        "--- Minecraft 皮肤渲染插件帮助 ---\n\n"
        "【指令1】/skin <param1> [param2]\n"
        "用法1 (推荐): /skin <渲染类型> <玩家名称>\n"
        "  » 示例: /skin walking Notch\n\n"
        f"  <渲染类型>: 可选。默认为 '{DEFAULT_RENDERTYPE}'。\n\n"
        "--- 所有可用的 [rendertype] 列表 ---\n"
        "用法2: /skin <玩家名称>\n"
        "  » 示例: /skin Notch\n"
        f"  <玩家名称>: 必需。玩家的 Minecraft ID。\n"
        + ", ".join(sorted(VALID_RENDERTYPES)) +
        "\n\n【指令2】/wallpaper [param1] [param2] ...\n"
        "用法1 (推荐): /wallpaper <壁纸ID> <玩家1> [玩家2] ...\n"
        "  » 示例: /wallpaper quick_hide Notch\n\n"
        "用法2: /wallpaper <玩家1> [玩家2] ...\n"
        "  » 示例: /wallpaper Notch Steve\n"
        f"  <壁纸ID>: 可选。默认为 '{DEFAULT_WALLPAPER}'。\n"
        "  <玩家...>: 必需。至少1个玩家名称。\n\n"
        "--- 可用的壁纸ID及玩家上限 ---\n" +
        "\n".join([f"  • {wp_id} (最多 {max_p} 个玩家)" for wp_id, max_p in sorted(WALLPAPER_CONFIGS.items())]) +
        "\n\n【指令3】/customskin <玩家名称> [相机预设] [焦点预设]\n"
        "  » 示例: /customskin Notch\n"
        "  使用自定义 .obj 模型文件渲染皮肤\n\n"
        "【指令4】/passport <渲染类型> <玩家名> <RGB> <称号> <想说的话> <愿望>\n"
        "  » 示例: /passport default Notch 255,0,0 my^title Hello,^world! 好运^连连Good-luck!\n\n"
        "  <渲染类型>: 皮肤渲染类型\n\n"
        "  <玩家名>: Minecraft 玩家名称\n\n"
        "  <RGB>: 颜色值，格式: 255,0,0\n\n"
        "  <称号>: 玩家称号（支持中英文，空格用^<shift+6>代替）\n\n"
        "  <想说的话>: 玩家想说的话（支持中英文，空格用^<shift+6>代替）\n\n"
        "  <愿望>: 玩家的愿望（支持中英文，空格用^<shift+6>代替）\n\n"
    )

def get_customskin_help_text():
    """获取自定义皮肤帮助文本"""
    return (
        "--- 自定义皮肤渲染帮助 ---\n\n"
        "用法: /customskin <玩家名称> [相机预设] [焦点预设]\n\n"
        "参数说明:\n"
        "  <玩家名称>: Minecraft 玩家名称\n"
        "  [相机预设]: 相机位置预设 (可选)\n"
        "  [焦点预设]: 相机焦点预设 (可选)\n\n"
        "使用流程:\n"
        "1. 输入命令: /customskin <玩家名称>\n"
        "2. 在指定时间内发送 .obj 模型文件\n"
        "3. 系统会自动处理并返回渲染结果\n\n"
        "支持的模型格式: .obj 文件"
    )

# ========== 文件传输功能 ==========
async def upload_to_tmpfiles(session, file_path: str) -> str | None:
    """上传文件到 tmpfiles.org 并返回下载链接"""
    try:
        with open(file_path, 'rb') as f:
            form_data = aiohttp.FormData()
            form_data.add_field('file', f, filename=os.path.basename(file_path))
            
            async with session.post('https://tmpfiles.org/api/v1/upload', data=form_data) as response:
                if response.status == 200:
                    result = await response.json()
                    if result.get('status') == 'success':
                        download_url = result['data']['url']
                        # 将 URL 转换为直接下载链接
                        if 'tmpfiles.org/' in download_url:
                            download_url = download_url.replace('tmpfiles.org/', 'tmpfiles.org/dl/')
                        return download_url
        return None
    except Exception as e:
        logger.error(f"上传到 tmpfiles.org 失败: {e}")
        return None

# ========== 主插件类 ==========
@register(
    "MCSkinRender",
    "SatellIta",
    "使用 Starlight API 异步获取 Minecraft 皮肤的多种渲染图、动作和护照生成",
    "1.1.1",
    "https://github.com/SatellIta/astrbot_plugin_minecraft_skin_render"
)
class MCSkinPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig = None):
        super().__init__(context)
        self.config = config
        self.session = aiohttp.ClientSession()
        self.plugin_path = pathlib.Path(__file__).parent

    def _load_font(self, font_names, size):
        """加载字体"""
        for font_name in font_names:
            font_path = self.plugin_path / font_name
            if font_path.exists():
                try:
                    return ImageFont.truetype(str(font_path), size)
                except Exception:
                    continue
        # 回退到默认字体
        return ImageFont.load_default()

    # ========== Passport 生成功能 ==========
    async def generate_passport(self, rendertype: str, username: str, rgb_str: str, 
                              title: str, message: str, wish: str) -> tuple[list[BytesIO], str | None]:
        """生成护照图片"""
        try:
            # 解析RGB
            theme_rgb = parse_theme_rgb(rgb_str)
            
            # 将连字符替换为空格（允许中文输入）
            # 在 generate_passport 函数中，将连字符替换为空格的部分改为：
            title = replace_hyphens_with_spaces(str(title) if title is not None else "")
            message = replace_hyphens_with_spaces(str(message) if message is not None else "")
            wish = replace_hyphens_with_spaces(str(wish) if wish is not None else "")
            
            # 获取玩家UUID和皮肤渲染
            uuid, error_msg = await get_player_uuid(self.session, username)
            if error_msg:
                return [], error_msg

            # 下载皮肤渲染图
            rendercrop = SKIN_RENDERCROP if rendertype == "skin" else DEFAULT_RENDERCROP
            render_url = STARLIGHT_RENDER_URL.format(
                rendertype=rendertype,
                uuid=uuid,
                rendercrop=rendercrop
            )
            
            async with self.session.get(render_url) as response:
                if response.status != 200:
                    return [], f"无法下载皮肤渲染图 (状态: {response.status})"
                skin_data = await response.read()
            
            # 加载所有图层
            skin_img = Image.open(BytesIO(skin_data)).convert("RGBA")
            base_img = Image.open(self.plugin_path / "base.png").convert("RGBA")
            base0_img = Image.open(self.plugin_path / "base0.png").convert("RGBA")
            overlay_img = Image.open(self.plugin_path / "overlay.png").convert("RGBA")
            
            # 调整所有图片到目标尺寸
            target_size = PASSPORT_CONFIG['output_size']
            
            # ========== 按比例缩放皮肤渲染图 ==========
            # 1. 计算缩放比例（保持宽高比）
            skin_ratio = skin_img.width / skin_img.height
            target_ratio = target_size[0] / target_size[1]
            
            if skin_ratio > target_ratio:
                # 原图更宽，按宽度缩放
                new_width = target_size[0]
                new_height = int(target_size[0] / skin_ratio)
            else:
                # 原图更高，按高度缩放
                new_height = target_size[1]
                new_width = int(target_size[1] * skin_ratio)
            
            # 2. 按比例缩放皮肤图
            skin_img_resized = skin_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # 3. 创建新的画布并居中放置缩放后的皮肤图
            skin_img_final = Image.new("RGBA", target_size, (0, 0, 0, 0))
            x_offset = (target_size[0] - new_width) // 2
            y_offset = (target_size[1] - new_height) // 2
            skin_img_final.paste(skin_img_resized, (x_offset, y_offset))
            
            # 使用按比例缩放后的皮肤图
            skin_img = skin_img_final
            
            # 其他图层仍然拉伸到目标尺寸
            base_img = base_img.resize(target_size, Image.Resampling.LANCZOS)
            base0_img = base0_img.resize(target_size, Image.Resampling.LANCZOS)
            overlay_img = overlay_img.resize(target_size, Image.Resampling.LANCZOS)
            
            # 1. 生成着色后的base0图
            base0_arr = np.array(base0_img)
            base0_tinted_arr = gray_to_hsv_tint(base0_arr, theme_rgb)
            base0_tinted = Image.fromarray(base0_tinted_arr, 'RGBA')
            
            # 2. 生成着色后的base与overlay叠加图
            base_arr = np.array(base_img)
            base_tinted_arr = gray_to_hsv_tint(base_arr, theme_rgb)
            base_tinted = Image.fromarray(base_tinted_arr, 'RGBA')
            
            # 在overlay上绘制文本
            overlay_draw = ImageDraw.Draw(overlay_img)
            W, H = target_size
            
            # 文本映射（直接使用原始文本，不再过滤不兼容字符）
            text_map = {
                'uid': username.upper(),
                'title': title,
                'message': message,
                'wish': wish,
                'watermark': PASSPORT_CONFIG['watermark_text']
            }
            
            for key, cfg in PASSPORT_CONFIG['text_config'].items():
                text = text_map[key]
                if not text:
                    continue

                cx = int(W * cfg['center'][0]); cy = int(H * cfg['center'][1])
                font = self._load_font(cfg['fonts'], cfg['size'])
                bbox = overlay_draw.textbbox((0,0), text, font=font)
                w = bbox[2] - bbox[0]; h = bbox[3] - bbox[1]

                if cfg['align'] == 'L':
                    if key == 'wish':
                        x = cx; y = cy - h
                        draw_text_with_wrap_and_left_bottom_align(overlay_draw, font, text, x, y, cfg['max_width'], cfg['opacity'])
                    else:
                        x = cx; y = cy - h//2
                        if cfg.get('shadow',0) > 0:
                            sd = cfg['shadow']
                            overlay_draw.text((x+sd, y+sd), text, font=font, fill=(0,0,0,cfg['opacity']))
                        overlay_draw.text((x, y), text, font=font, fill=(255,255,255,cfg['opacity']))

                elif cfg['align'] == 'R':
                    if key == 'uid':
                        size = cfg['size']
                        while w > PASSPORT_CONFIG['uid_max_length'] and size > 8:
                            size -= 2
                            font = self._load_font(cfg['fonts'], size)
                            bbox = overlay_draw.textbbox((0,0), text, font=font)
                            w = bbox[2] - bbox[0]; h = bbox[3] - bbox[1]

                    x = cx - w; y = cy - h//2
                    if cfg.get('shadow',0) > 0:
                        sd = cfg['shadow']
                        overlay_draw.text((x+sd, y+sd), text, font=font, fill=(0,0,0,cfg['opacity']))
                    overlay_draw.text((x, y), text, font=font, fill=(255,255,255,cfg['opacity']))

                else:  # 'M' center
                    draw_centered_text(overlay_draw, font, text, cx, cy, cfg['opacity'])
            
            # 叠加base和overlay
            base_overlay = Image.alpha_composite(base_tinted, overlay_img)
            
            # 3. 生成完整成品图 (skin + base_overlay + base0_tinted)
            result = Image.alpha_composite(skin_img, base0_tinted)
            result = Image.alpha_composite(result, base_overlay)
            
            # 转换为BytesIO
            output_images = []
            for img, name in [
                (result, "成品图"),
                (base0_tinted, "顶端着色图"),
                (base_overlay, "底部叠加图"),
                (skin_img, "皮肤渲染图")
            ]:
                bio = BytesIO()
                img.save(bio, format='PNG')
                bio.seek(0)
                output_images.append(bio)
            
            return output_images, None
            
        except Exception as e:
            logger.error(f"生成通行证时发生错误: {e}", exc_info=True)
            return [], f"生成通行证时发生错误: {str(e)}"

    # ========== 命令处理器 ==========
    @filter.command("skin")
    async def get_skin(
        self,
        event: AstrMessageEvent,
        param1: str = None,
        param2: str = None
    ):
        """获取 Minecraft 玩家皮肤的渲染图"""
        if not param1:
            yield event.plain_result(
                "错误：请提供玩家名称。\n"
                "用法1: /skin <玩家名称>\n"
                "用法2: /skin <渲染类型> <玩家名称>"
            )
            return

        username: str
        rendertype: str

        if param2:
            is_valid_type, _ = validate_rendertype(param1.lower())
            if is_valid_type:
                rendertype = param1
                username = param2
            else:
                username = param1
                rendertype = DEFAULT_RENDERTYPE
                logger.warning(f"参数 '{param1}' 不是有效的渲染类型，已忽略第二个参数 '{param2}'，并使用默认渲染类型。")
        else:
            username = param1
            rendertype = DEFAULT_RENDERTYPE

        result = await process_skin_command(self.session, username, rendertype)
        if isinstance(result, str):
            yield event.plain_result(result)
        else:
            yield event.chain_result(result)

    @filter.command("wallpaper")
    async def get_wallpaper(
        self,
        event: AstrMessageEvent,
        param1: str = None,
        param2: str = None,
        param3: str = None,
        param4: str = None
    ):
        """获取 Minecraft 壁纸"""
        if not param1:
            yield event.plain_result(
                "错误：请提供玩家名称或壁纸ID。\n"
                f"用法1: /wallpaper <玩家名称1> [玩家名称2] ...\n"
                f"用法2: /wallpaper <壁纸ID> <玩家名称1> [玩家名称2] ..."
            )
            return

        wallpaper_id: str
        usernames: list[str]

        is_valid_wallpaper, _, _ = validate_wallpaper(param1.lower())
        if is_valid_wallpaper:
            wallpaper_id = param1
            usernames = [p for p in [param2, param3, param4] if p]
        else:
            wallpaper_id = DEFAULT_WALLPAPER
            usernames = [p for p in [param1, param2, param3, param4] if p]

        result = await process_wallpaper_command(self.session, wallpaper_id, usernames)
        if isinstance(result, str):
            yield event.plain_result(result)
        else:
            yield event.chain_result(result)

    @filter.command("skinhelp")
    async def skin_help(self, event: AstrMessageEvent):
        """显示帮助信息"""
        full_help = get_help_text()
        yield event.plain_result(full_help)

    @filter.command("customskinhelp")
    async def custom_skin_help(self, event: AstrMessageEvent):
        """显示自定义皮肤帮助"""
        full_help = get_customskin_help_text()
        yield event.plain_result(full_help)

    @filter.command("customskin")
    async def custom_skin(self, event: AstrMessageEvent, username: str = None, camera_preset: str = None, focal_preset: str = None):
        """使用自定义模型渲染皮肤"""
        if not username:
            yield event.plain_result("错误：请提供玩家名称。\n用法: /customskin <玩家名称> [相机预设] [焦点预设]")
            return

        # 1. 获取玩家 UUID
        uuid, error_msg = await get_player_uuid(self.session, username)
        if error_msg:
            yield event.plain_result(error_msg)
            return

        # 2. 发送提示
        prompt_msg = (
            f"请在 {FILE_WAIT_TIMEOUT} 秒内发送一个 .obj 模型文件 "
            f"来为玩家 {username} 进行渲染。"
        )
        await event.send(event.plain_result(prompt_msg))

        # 2.1 解析用户可选参数
        camera_param_raw = camera_preset
        focal_param_raw = focal_preset

        # 3. 定义并启动会话控制器
        @session_waiter(timeout=FILE_WAIT_TIMEOUT, record_history_chains=False)
        async def custom_skin_waiter(controller: SessionController, event: AstrMessageEvent):
            file_component = None
            # 遍历消息组件，查找 File 类型的组件
            for component in event.get_messages():
                if isinstance(component, File):
                    file_component = component
                    break
            
            if not file_component:
                return

            local_path = await file_component.get_file()

            try:
                # 根据配置决定使用本地文件服务还是公共中转服务
                if self.config and self.config.get("use_file_transfer"):
                    # 使用公共中转服务 (tmpfiles.org)
                    logger.info("use_file_transfer 已开启，使用 tmpfiles.org 上传...")
                    stable_url = await upload_to_tmpfiles(self.session, local_path)
                else:
                    # 使用内置文件服务
                    logger.info("use_file_transfer 未开启或未配置，使用内置文件服务注册...")
                    stable_url = await file_component.register_to_file_service()

                if not stable_url:
                    await event.send(event.plain_result("错误：文件上传或注册失败，无法获取有效的 URL。"))
                    controller.stop()
                    return

                logger.info(f"文件服务返回的稳定 URL: {stable_url}")

                # 2. 解析并决定最终使用的相机与焦点参数
                def resolve_position_param(raw_value, presets, default):
                    if not raw_value:
                        return default
                    raw_lower = raw_value.lower()
                    # 如果用户提供了一个预置名
                    if raw_lower in presets:
                        return presets[raw_lower]
                    # 否则尝试解析 JSON
                    try:
                        parsed = json.loads(raw_value)
                        if isinstance(parsed, dict):
                            return parsed
                    except Exception:
                        pass
                    # 无法解析则返回默认并告知用户（但不抛错）
                    return default

                camera_position = resolve_position_param(camera_param_raw, CAMERA_PRESETS, DEFAULT_CAMERA_POSITION)
                camera_focal = resolve_position_param(focal_param_raw, FOCAL_PRESETS, DEFAULT_CAMERA_FOCAL_POINT)

                # 3. 将 URL 和额外参数传递给 action 构建最终的渲染URL
                result_chain = await upload_and_render_custom_skin(
                    self.session,
                    uuid,
                    stable_url,
                    username,
                    camera_position=camera_position,
                    camera_focal_point=camera_focal,
                )

                # 4. 发送结果
                await event.send(event.chain_result(result_chain))
                controller.stop() # 成功处理，结束会话

            except Exception as e:
                logger.error(f"文件服务注册或处理时失败: {e}", exc_info=True)
                await event.send(event.plain_result("错误：文件处理失败。请检查机器人配置文件中的 `callback_api_base` 是否正确设置。"))
                controller.stop()

            finally:
                if local_path and os.path.exists(local_path):
                    # 定义一个后台清理函数
                    async def delayed_cleanup(path, delay=20):
                        await asyncio.sleep(delay)
                        try:
                            if os.path.exists(path):
                                os.remove(path)
                                logger.info(f"延迟清理完成: {path}")
                            else:
                                logger.warning(f"文件已不存在，跳过清理：{path}")
                        except Exception as e:
                            logger.error(f"延迟清理文件{path}时失败：{e}")

                    logger.info(f"为{local_path}创建了延迟清理任务")
                    asyncio.create_task(delayed_cleanup(local_path))

        try:
            await custom_skin_waiter(event)
        except TimeoutError:
            yield event.plain_result("操作超时，已取消渲染。")
        except Exception as e:
            logger.error(f"customskin 会话期间发生未知错误: {e}", exc_info=True)
            yield event.plain_result(f"处理过程中发生内部错误: {e}")
        finally:
            event.stop_event()

    @filter.command("passport")
    async def get_passport(
        self,
        event: AstrMessageEvent,
        rendertype: str = None,
        username: str = None,
        rgb: str = None,
        title: str = None,
        message: str = None,
        wish: str = None
    ):
        """
        生成Minecraft通行证
        用法: /passport <渲染类型> <玩家名> <RGB> <称号> <想说的话> <愿望>
        示例: /passport default Notch "255,0,0" "我的-称号" "你好-世界" "好运-连连"
        """
        if not all([rendertype, username, rgb, title, message, wish]):
            yield event.plain_result(
                "错误：请提供所有必需参数。\n"
                "用法: /passport <渲染类型> <玩家名> <RGB> <称号> <想说的话> <愿望>\n"
                "示例: /passport default Notch \"255,0,0\" \"我的-称号\" \"你好-世界\" \"好运-连连\"\n"
                "注意: 参数中的空格请用连字符(-)代替"
            )
            return

        # 验证渲染类型
        is_valid, error_msg = validate_rendertype(rendertype.lower())
        if not is_valid:
            yield event.plain_result(error_msg)
            return

        yield event.plain_result("正在生成通行证，请稍候...")

        # 生成护照
        images, error_msg = await self.generate_passport(rendertype, username, rgb, title, message, wish)
        if error_msg:
            yield event.plain_result(error_msg)
            return

        # 发送所有图片
        image_names = ["完整成品图", "顶端着色图", "底部叠加图", "皮肤渲染图"]
        chain = [Comp.Plain(f"为 {username} 生成的通行证图片：\n")]
        
        for i, (img_bio, name) in enumerate(zip(images, image_names)):
            chain.append(Comp.Plain(f"\n{name}：\n"))
            chain.append(Comp.Image.fromBytes(img_bio.getvalue()))
        
        yield event.chain_result(chain)

    async def terminate(self):
        """清理资源"""
        await self.session.close()
        logger.info("MCSkinPlugin: aiohttp session 已成功关闭")
