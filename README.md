# DouyinStreamOps MVP

抖音直播采集与回放分析的 Windows 桌面端 + 云端 SaaS MVP。

当前仓库包含两条主链路：

- 直播采集链路：采集直播流、人数曲线、弹幕，并落地到本地文件
- 回放分析链路：从回放视频和结构化文件生成时间线、摘要和热点分析结果

## 1. 技术栈

- 桌面端：`PySide6 + Python + ffmpeg + faster-whisper`
- 本地状态：`SQLite`
- 直播采集：`f2`
- 说话人区分：`pyannote.audio`（可选）
- 云端：`FastAPI + RQ + PostgreSQL`
- 报告输出：`Markdown -> HTML`
- 打包：`Nuitka`

## 2. 目录结构

- `src/post_live_analyst`：回放分析核心
- `src/desktop_client`：Windows 桌面客户端
- `src/cloud_api`：云端 API 与异步任务
- `scripts/launch_desktop.py`：启动综合桌面端
- `scripts/launch_capture_desktop.py`：启动采集监控端
- `scripts/launch_analyzer_desktop.py`：启动视频分析端
- `scripts/start_api.py`：启动 FastAPI
- `scripts/start_worker.py`：启动 RQ Worker
- `scripts/download_whisper_model.py`：预下载 faster-whisper 模型

## 3. 功能概览

### 3.1 本地分析链路

1. 导入 `replay.mp4 / replay.flv`
2. 导入 `occupant_history.csv`
3. 可选导入 `danmaku_history.jsonl / csv`
4. 使用 `ffmpeg` 抽取 `voicetrack.wav`
5. 使用 `faster-whisper` 生成 `transcript.json`
6. 可选使用 `pyannote.audio` 给 transcript 回填 `speaker_id`
7. 对齐人数、转写、弹幕，生成 `detailed_timeline.json`
8. 调用 OpenAI-compatible 模型服务做热点分析
9. 上传结构化结果到云端

### 3.2 直播采集链路

桌面端支持以 `room_id / webcast_id / 直播链接 / 主页链接` 为入口采集直播资产：

1. 使用 `f2` 获取直播间信息
2. 下载直播流并落地为本地 `flv/mp4`
3. 轮询直播人数并生成 `occupant_history.csv`
4. 订阅弹幕并生成 `danmaku_history.jsonl`
5. 采集完成后回填到桌面端分析输入区域

## 4. 环境要求

建议环境：

- Windows 10 / 11
- Python 3.11 或 3.12
- 可用的 `ffmpeg`
- 如果启用完整云端联调：PostgreSQL + Redis

建议先确认：

```powershell
python --version
ffmpeg -version
```

## 5. 预装依赖

### 5.1 桌面端 Python 依赖

```powershell
pip install -r requirements-desktop.txt
```

桌面端依赖包括：

- `PySide6`
- `faster-whisper`
- `f2`
- `pyannote.audio`
- `requests / httpx / pandas`

### 5.2 云端 Python 依赖

```powershell
pip install -r requirements-server.txt
```

### 5.3 ffmpeg

本项目依赖 `ffmpeg` 从回放视频中抽取音轨。不安装会导致：

- 无法生成 `voicetrack.wav`
- ASR 无法执行

如果 `ffmpeg` 不在系统 `PATH` 中，可以通过环境变量指定：

```powershell
$env:FFMPEG_BINARY="D:\tools\ffmpeg\bin\ffmpeg.exe"
```

### 5.4 f2

直播采集依赖 `f2`。如果 `f2` 不可用：

- 无法采集直播流
- 无法获取直播间人数和弹幕

桌面端会在运行时给出提醒，但采集链路无法正常工作。

## 6. 模型准备

### 6.1 faster-whisper 模型

本项目的 ASR 默认使用 `faster-whisper`。

模型引用顺序如下：

1. `DSO_ASR_MODEL`
2. `FASTER_WHISPER_MODEL`
3. 项目目录下的本地模型目录
4. 最后回退为 `small`

当前代码会优先尝试下面两个本地目录：

- `models/faster-whisper-small`
- `models/Systran--faster-whisper-small`

如果这些目录都不存在，就会退回到 `small`，由 `faster-whisper` 自己从 Hugging Face 缓存加载或下载。

### 6.2 预下载 whisper 模型

建议首次部署时主动预下载模型，避免第一次分析时临时拉取。

```powershell
$env:HF_TOKEN="你的 Hugging Face Token"
python scripts/download_whisper_model.py --model small --local-dir models/faster-whisper-small
```

下载完成后，桌面端会优先加载：

- `D:\privacy\ai_coder\DouyinStreamOps\models\faster-whisper-small`

如果你想强制使用某个目录，也可以直接指定：

```powershell
$env:DSO_ASR_MODEL="D:\privacy\ai_coder\DouyinStreamOps\models\faster-whisper-small"
```

### 6.3 pyannote 说话人区分模型

`pyannote.audio` 不是必需项，只在你希望 transcript 带 `speaker_id` 时启用。

默认模型为：

- `pyannote/speaker-diarization-3.1`

启用前需要至少配置一个 token：

```powershell
$env:PYANNOTE_AUTH_TOKEN="你的 Hugging Face Token"
```

也兼容：

- `HF_TOKEN`
- `HUGGINGFACE_HUB_TOKEN`

如果没有 token 或没有安装 `pyannote.audio`，分析流程不会报死，只会跳过 diarization，并在诊断信息里说明原因。

## 7. 环境变量

### 7.1 桌面端常用环境变量

```powershell
$env:SERVER_BASE_URL="http://127.0.0.1:8000"
$env:OPENAI_BASE_URL="http://127.0.0.1:8317/v1"
$env:OPENAI_API_KEY="your-api-key-1"
$env:OPENAI_CHAT_MODEL_ID="gpt-5.4"
$env:DEEPSEEK_API_KEY="..."
$env:DEEPSEEK_MODEL_ID="deepseek-v4-pro"
$env:DOUYIN_COOKIE="..."
$env:DOUYIN_USER_AGENT="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"
$env:DSO_ASR_MODEL="D:\privacy\ai_coder\DouyinStreamOps\models\faster-whisper-small"
$env:PYANNOTE_AUTH_TOKEN="..."
```

说明：

- `SERVER_BASE_URL`
  - 云端 API 地址
- `OPENAI_BASE_URL`
  - OpenAI-compatible 模型代理地址
- `OPENAI_API_KEY`
  - 模型代理鉴权
- `OPENAI_CHAT_MODEL_ID`
  - 热点分析使用的模型名
- `DEEPSEEK_API_KEY`
  - 分析端直连 DeepSeek 时使用的 API Key
- `DEEPSEEK_MODEL_ID`
  - 分析端直连 DeepSeek 时使用的模型名，默认可用 `deepseek-v4-pro`
- `DOUYIN_COOKIE`
  - 直播采集建议必配，否则采集可能失败或不稳定
- `DSO_ASR_MODEL`
  - 可选。显式指定本地 whisper 模型目录
- `PYANNOTE_AUTH_TOKEN`
  - 可选。启用 speaker diarization

### 7.2 云端环境变量

```powershell
$env:DATABASE_URL="postgresql+psycopg://postgres:postgres@127.0.0.1:5432/douyin_stream_ops"
$env:REDIS_URL="redis://127.0.0.1:6379/0"
```

## 8. 启动方式

### 8.1 仅运行桌面端

适用于：

- 本地录播采集
- 本地回放分析
- 联调本地 OpenAI-compatible 代理

```powershell
pip install -r requirements-desktop.txt
python scripts/launch_desktop.py
```

### 8.2 启动采集监控端

适用于：

- 无人值守监控直播间
- 自动落地录播、人数和弹幕文件
- 手动同步抖音关注列表到监控列表
- 内存不足时通过飞书 / 钉钉机器人发告警

```powershell
pip install -r requirements-desktop.txt
python scripts/launch_capture_desktop.py
```

说明：

- 采集端会自动读取已保存的监控列表
- 如果存在已启用的主页，启动后会自动进入监控
- “同步关注列表”需要填写关注来源账号的抖音主页链接或 `sec_uid`
- 关注同步项会随下次手动同步自动增删，手动添加项不会被取消关注联动删除
- 机器人告警配置在采集端界面内保存

### 8.3 启动视频分析端

适用于：

- 导入采集得到的回放和结构化文件
- 透传到 OpenAI-compatible 模型或 DeepSeek 做热点分析
- 上传结构化结果到云端

```powershell
pip install -r requirements-desktop.txt
python scripts/launch_analyzer_desktop.py
```

说明：

- 分析端内置 `OpenAI 兼容代理` 和 `DeepSeek 直连` 两种模型配置
- 切到 `DeepSeek 直连` 后，默认接口地址为 `https://api.deepseek.com`

### 8.4 运行完整联调环境

适用于：

- 桌面端上传结构化结果到云端
- 云端 API / Worker 联调

终端 1：

```powershell
pip install -r requirements-server.txt
python scripts/start_api.py
```

终端 2：

```powershell
pip install -r requirements-server.txt
python scripts/start_worker.py
```

终端 3：

```powershell
pip install -r requirements-desktop.txt
python scripts/launch_analyzer_desktop.py
```

## 9. 推荐的首次启动顺序

建议在新机器上按这个顺序准备：

1. 安装 Python 3.11 / 3.12
2. 安装 `ffmpeg`
3. `pip install -r requirements-desktop.txt`
4. 配置 `DOUYIN_COOKIE`
5. 配置本地 OpenAI-compatible 代理服务
6. 预下载 `faster-whisper` 模型
7. 如需说话人区分，再配置 `PYANNOTE_AUTH_TOKEN`
8. 启动桌面端
9. 如需联调云端，再启动 API 和 Worker

## 10. 常见问题

### 10.1 没有 `ffmpeg`

现象：

- 无法生成 `voicetrack.wav`
- ASR 不执行

处理：

- 安装 `ffmpeg`
- 或通过 `FFMPEG_BINARY` 指向 `ffmpeg.exe`

### 10.2 没有预下载 whisper 模型

现象：

- 第一次运行转写时会临时下载模型
- 启动分析可能明显变慢

处理：

- 提前运行 `scripts/download_whisper_model.py`
- 或显式设置 `DSO_ASR_MODEL`

### 10.3 没有 `PYANNOTE_AUTH_TOKEN`

现象：

- transcript 仍可生成
- 但不会有 `speaker_id`

处理：

- 配置 `PYANNOTE_AUTH_TOKEN`
- 或接受跳过 diarization

### 10.4 没有 `DOUYIN_COOKIE`

现象：

- 直播采集失败
- 直播间状态识别不稳定

处理：

- 补充有效的抖音 Cookie

### 10.5 没有 OpenAI-compatible 模型服务

现象：

- 时间线仍可生成
- 但热点分析会失败或为空

处理：

- 启动本地代理服务
- 检查 `OPENAI_BASE_URL / OPENAI_API_KEY / OPENAI_CHAT_MODEL_ID`

### 10.6 `f2` 不可用

现象：

- 无法采集直播流
- 无法抓取人数与弹幕

处理：

- 确认 `f2` 已安装且版本可用

## 11. Nuitka 打包示例

```powershell
python -m nuitka ^
  --standalone ^
  --enable-plugin=pyside6 ^
  --include-package=desktop_client ^
  --include-package=post_live_analyst ^
  --output-dir=build ^
  scripts/launch_desktop.py
```

## 12. 当前版本说明

- 当前仓库优先覆盖 MVP 主链路
- 对象存储与更完整的付费热点分析流程仍是后续扩展项
- 直播采集、转写、时间线分析、模型热点分析和云端上传已具备基础串联能力
