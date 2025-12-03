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

# ========== 渲染类型配置 ==========
VALID_RENDERTYPES = {
    "default", "marching", "walking", "crouching", "crossed", "criss_cross",
    "ultimate", "isometric", "head", "custom", "cheering", "relaxing",
    "trudging", "cowering", "pointing", "lunging", "dungeons", "facepalm",
    "sleeping", "dead", "archer", "kicking", "mojavatar", "reading",
    "high_ground", "clown", "bitzel", "pixel", "ornament", "skin", "profile"
}

# ========== 默认配置 ==========
DEFAULT_RENDERTYPE = "default"
DEFAULT_RENDERCROP = "full"
SKIN_RENDERCROP = "default"

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
            'size': 78,
            'max_width': 100000000,
            'fonts': ['NotoSansSChineseMedium-7.ttf'],
            'opacity': 230,
            'shadow': 2
        },
        'message': {
            'center': (0.5, 0.82),
            'align': 'M',
            'size': 70,
            'max_width': 100000000,
            'fonts': ['NotoSansSChineseMedium-7.ttf'],
            'opacity': 200,
            'shadow': 0
        },
        'wish': {
            'center': (0.16, 0.946),
            'align': 'L',
            'size': 60,
            'max_width': 5000000000,
            'fonts': ['NotoSansSChineseMedium-7.ttf'],
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
                    f"输入 /producthelp 查看完整列表。")
        return False, error_msg
    return True, None

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

def replace_hyphens_with_spaces(text: str) -> str:
    """将文本中的^替换为空格，允许中文输入"""
    if text is None:
        return ""
    return str(text).replace('^', ' ')

def get_product_help_text():
    """获取文创渲染帮助文本"""
    return (
        "--- Minecraft 文创渲染插件帮助 ---\n"
        "【指令1】/passport <渲染类型> <玩家名> <RGB> <称号> <想说的话> <愿望>\n"
        "  » 示例: /passport default Notch 255,0,0 my^title Hello,^world! 好运^连连Good-luck!\n"
        "  参数说明:\n"
        "  <渲染类型>: 皮肤渲染类型，可选值如下：\n"
        f"    {', '.join(VALID_RENDERTYPES)}\n"
        "  <玩家名>: Minecraft 玩家名称\n"
        "  <RGB>: 颜色值，格式: 255,0,0 (红,绿,蓝)\n"
        "  <称号>: 玩家称号（支持中英文，空格用^<shift+6>代替）\n"
        "  <想说的话>: 玩家想说的话（支持中英文，空格用^<shift+6>代替）\n"
        "  <愿望>: 玩家的愿望（支持中英文，空格用^<shift+6>代替）\n"
        "  注意：所有参数都是必需的，请确保提供完整的6个参数。\n"
        "  连字符(^)在显示时会自动转换为空格。\n"
    )

# ========== 主插件类 ==========
@register(
    "MCProductRenderer",
    "CecilyGao & SatellIta",
    "生成 Minecraft 玩家文创预览图",
    "1.0.0",
    "https://github.com/CecilyGao/astrbot_plugin_minecraft_product_render"
)
class MCProductPlugin(Star):
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
        """生成通行证图片"""
        try:
            # 解析RGB
            theme_rgb = parse_theme_rgb(rgb_str)
            
            # 将连字符替换为空格（允许中文输入）
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
            logger.error(f"生成文创时发生错误: {e}", exc_info=True)
            return [], f"生成文创时发生错误: {str(e)}"

    # ========== 命令处理器 ==========
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
        示例: /passport default Notch "255,0,0" "我的^称号" "你好^世界" "好运^连连"
        """
        if not all([rendertype, username, rgb, title, message, wish]):
            yield event.plain_result(
                "错误：请提供所有必需参数。\n\n"
                "用法: /passport <渲染类型> <玩家名> <RGB> <称号> <想说的话> <愿望>\n\n"
                "示例: /passport default Notch 255,0,0 我的^称号 你好^世界 好运^连连\n\n"
                "注意: 参数中的空格请用尖号(^)代替\n\n"
                "输入 /producthelp 查看详细帮助"
            )
            return

        # 验证渲染类型
        is_valid, error_msg = validate_rendertype(rendertype.lower())
        if not is_valid:
            yield event.plain_result(error_msg)
            return

        yield event.plain_result("正在生成通行证，请稍候...")

        # 生成通行证
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

    @filter.command("producthelp")
    async def product_help(self, event: AstrMessageEvent):
        """显示帮助信息"""
        full_help = get_product_help_text()
        yield event.plain_result(full_help)

    async def terminate(self):
        """清理资源"""
        await self.session.close()
        logger.info("MCProductPlugin: aiohttp session 已成功关闭")
