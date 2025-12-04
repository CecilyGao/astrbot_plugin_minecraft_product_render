# AstrBot Minecraft 文创渲染插件 (MCProductRenderer)
用于获取 Minecraft 玩家皮肤的 3D 渲染图（支持动作）用于制作通行证、明信片等MC文创，本仓库素材为南京大学MC社使用，如有需求可替换仓库文件。

# 🔧 安装
方法一：使用插件市场 (推荐)

搜索 MC文创渲染插件 并安装

方法二：Git Clone

进入 AstrBot 的 data/plugins/ 目录，然后执行：

```bash
git clone https://github.com/CecilyGao/astrbot_plugin_minecraft_product_render
```

安装依赖

无论使用哪种方法，插件的依赖都会在机器人下次重启时自动安装。

# 🚀 使用说明

## 指令1：获取通行证渲染
`/passport [rendertype] <username> RGB 'title' 'message' 'wish' `

### 参数（参数内空格使用^占位，参数间使用空格隔开）
- `[rendertype]`: 列表选择。渲染类型，默认为 `default`
- `<username>`: 必需。玩家名称（带空格请使用引号，如 "Steve Jobs"）
- `RGB`: 必需。输入RGB值选择通行证主色调
- `title`: 必需。玩家头衔，空格使用^占位
- `message`: 必需。玩家信息、格言、座右铭等，空格使用^占位
- `wish`: 必需。玩家愿望，空格使用^占位

### 示例
- `/passport default AintCecily 170,140,30 Ain't^A^Lord Curiosity^will^never^let^me^go. 我是奶龙！` - 默认全身渲染
- `/passport walking Noname2309 153,102,204 114514 Keep^the^original^heart^and-purity See-you-next-time` - 行走动作的全身渲染
- `/passport cheering AintCecily 120,140,30 Journal^Editor Curiosity^will^never^let^me^go. 我才是奶龙！` - 欢呼动作的全身渲染

<img width="827" height="2597" alt="template" src="https://github.com/user-attachments/assets/1bfb353b-e184-452e-ac19-e74a103c0856" />

---
## 指令2：获取明信片渲染
`/postcard <direction> <position> <color> <rotation>（同时引用一张图片）`

### 参数（参数内空格使用^占位，参数间使用空格隔开）
- `<direction>`: 必需。明信片方向，h为横向（3840×2160），v为竖向（2160×3840）
- `<position>`: 必需。水印位置，l为左下角，r为右下角
- `<color>`: 必需。水印颜色，b为黑色水印，w为白色水印
- `<rotation>`: 可选。原图旋转方向，+为顺时针旋转90°，-为逆时针旋转90°，0为不旋转，留空默认为0

### 示例
- `/postcard h l w（引用一张图片）` 
- `/postcard v r b -（引用一张图片）`

<img width="746" height="1411" alt="IMG_570" src="https://github.com/user-attachments/assets/a6b237e4-0f8f-4bec-b556-80781b66293e" />

---
## 帮助命令
`/producthelp` - 查看所有指令与可用的渲染类型
