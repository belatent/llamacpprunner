# llama.cpp Windows 启动器 GUI

一个面向 Windows 的 `llama.cpp` 可视化启动器，支持参数配置、配置方案保存、实时日志查看与一键启动/停止。

## 功能

- 左右分栏：左侧参数配置，右侧宽日志输出窗口
- 覆盖常见启动参数：路径/模型、服务、采样、性能、高级、推测解码
- 实时命令预览与参数校验
- 子进程启动 `llama-server`/`llama-cli`，实时捕获 `stdout/stderr`
- 配置方案读写：`config/profiles/*.json`
- 状态持久化：`config/state.json`

## 运行

```powershell
uv venv
uv pip install -r requirements.txt
uv run python -m app.main
```

## 绿色打包（便携版）

```powershell
pip install pyinstaller
pyinstaller --noconfirm --onefile --windowed --name llama-launcher app/main.py
```

生成文件位于 `dist/llama-launcher.exe`，可直接拷贝到目标机器运行。
