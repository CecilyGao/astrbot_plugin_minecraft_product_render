# AstrBot Minecraft 文创渲染插件 (MCProductRenderer)
用于获取 Minecraft 玩家皮肤的 3D 渲染图（支持动作）用于制作通行证等MC文创，本仓库限南京大学MC社使用，如有需求可替换仓库文件。

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

## 帮助命令
`/producthelp` - 查看所有可用的渲染类型和壁纸列表
