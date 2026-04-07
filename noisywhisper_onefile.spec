from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules

block_cipher = None

project_root = Path.cwd()

datas = []
binaries = []

for static_name in ("README.md", "NoisyWhisper_TechSpec_v1.0.md"):
    static_path = project_root / static_name
    if static_path.exists():
        datas.append((str(static_path), "."))

ffmpeg_exe = project_root / "ffmpeg.exe"
ffmpeg_bin = project_root / "ffmpeg"
if ffmpeg_exe.exists():
    binaries.append((str(ffmpeg_exe), "."))
elif ffmpeg_bin.exists():
    binaries.append((str(ffmpeg_bin), "."))

icon_path = project_root / "assets" / "icon.ico"
icon = str(icon_path) if icon_path.exists() else None

hiddenimports = []
hiddenimports += collect_submodules("faster_whisper")
hiddenimports += collect_submodules("ctranslate2")
hiddenimports += collect_submodules("silero_vad")
hiddenimports += collect_submodules("onnxruntime")
hiddenimports += collect_submodules("av")

binaries += collect_dynamic_libs("PyQt6")

datas += collect_data_files("tokenizers")
datas += collect_data_files("huggingface_hub")

a = Analysis(
    ["main.py"],
    pathex=[str(project_root)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="NoisyWhisper",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    icon=icon,
)
