# DouyinStreamOps MVP

Windows 客户端 + 云端 SaaS 的直播回放分析 MVP。

## 技术栈

- 客户端: `PySide6 + Python + ffmpeg + faster-whisper`
- 打包: `Nuitka`
- 本地状态: `SQLite`
- 云端: `FastAPI + RQ + PostgreSQL`
- 报告: `Markdown -> HTML`

## 目录

- `src/post_live_analyst`: 共享分析内核
- `src/desktop_client`: Windows 客户端
- `src/cloud_api`: 云端 API 与异步报告
- `scripts/start_api.py`: 启动 FastAPI
- `scripts/start_worker.py`: 启动 RQ Worker
- `scripts/launch_desktop.py`: 启动桌面端

## 本地处理链路

1. 导入 `replay.mp4`
2. 导入 `occupant_history.csv`
3. 可选导入 `danmaku_history.jsonl`
4. `ffmpeg` 分离 `voicetrack.wav`
5. `faster-whisper` 转写 `transcript.json`
6. 对齐人数、话术、弹幕并生成 `detailed_timeline.json`
7. 调用本地 `cli_proxy`
8. 上传结构化结果到云端

## 直播采集链路

客户端支持以 `room_id / webcast_id / 直播链接` 为入口采集直播资产：

1. 使用 `f2` 获取直播间信息
2. 下载直播流，落成本地 `flv/mp4`
3. 每秒轮询直播间人数，生成 `occupant_history.csv`
4. 订阅直播弹幕，生成 `danmaku_history.jsonl`
5. 采集完成后自动回填到桌面端分析表单

## 快速启动

### 服务端

```bash
pip install -r requirements-server.txt
python scripts/start_api.py
python scripts/start_worker.py
```

### 客户端

```bash
pip install -r requirements-desktop.txt
python scripts/launch_desktop.py
```

## 环境变量

```bash
SERVER_BASE_URL=http://127.0.0.1:8000
OPENAI_BASE_URL=http://127.0.0.1:8317/v1
OPENAI_API_KEY=your-api-key-1
OPENAI_CHAT_MODEL_ID=gpt-5.4
DOUYIN_COOKIE=
DOUYIN_USER_AGENT=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36
DATABASE_URL=postgresql+psycopg://postgres:postgres@127.0.0.1:5432/douyin_stream_ops
REDIS_URL=redis://127.0.0.1:6379/0
```

## Nuitka 打包示例

```bash
python -m nuitka ^
  --standalone ^
  --enable-plugin=pyside6 ^
  --include-package=desktop_client ^
  --include-package=post_live_analyst ^
  --output-dir=build ^
  scripts/launch_desktop.py
```

## 说明

- 当前版本优先打通 MVP 主链路。
- 对象存储仍保留为 `todo`。
- `DATABASE_URL` 支持 PostgreSQL，默认本地开发回退为 SQLite。
- 直播采集依赖 `f2`，当前环境如果未安装，客户端会提示后再继续。
