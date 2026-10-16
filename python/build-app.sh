#!/usr/bin/env bash
# 打包斗地主为 macOS .app —— 双击运行, 无需装 Python
# 产物: dist/斗地主.app  (PyInstaller onedir + 手写 Info.plist + launcher)
set -e
cd "$(dirname "$0")"
APP="${1:-斗地主.app}"
DIST="dist"
ROOT="$(pwd)"

echo "==> PyInstaller 打包 (onedir, 精简依赖)"
pyinstaller --noconfirm --clean \
  --name "斗地主" \
  --onedir \
  --console \
  --add-data "$ROOT/models:models" \
  --add-data "$ROOT/models_sl:models_sl" \
  --add-data "$ROOT/models_wp_backup:models_wp_backup" \
  --add-data "$ROOT/public:public" \
  --add-data "$ROOT/cards.py:." \
  --add-data "$ROOT/encoding.py:." \
  --add-data "$ROOT/ai_douzero.py:." \
  --add-data "$ROOT/env:env" \
  --hidden-import aiohttp \
  --hidden-import aiohttp.web \
  --hidden-import onnxruntime \
  --hidden-import onnxruntime.capi \
  --hidden-import numpy \
  --exclude-module torch \
  --exclude-module pyarrow \
  --exclude-module scipy \
  --exclude-module pandas \
  --exclude-module matplotlib \
  --exclude-module PIL \
  --exclude-module lxml \
  --exclude-module pydantic \
  --exclude-module IPython \
  --exclude-module jupyter \
  --exclude-module notebook \
  --exclude-module pytest \
  --exclude-module setuptools \
  --exclude-module tkinter \
  --exclude-module PyQt5 --exclude-module PyQt6 \
  --exclude-module PySide2 --exclude-module PySide6 \
  --exclude-module sympy --exclude-module networkx \
  --exclude-module sklearn --exclude-module tensorflow \
  --exclude-module jax --exclude-module keras \
  --exclude-module cv2 --exclude-module imageio \
  --exclude-module boto3 --exclude-module botocore \
  --exclude-module google --exclude-module grpc \
  --exclude-module sqlite3 --exclude-module sqlalchemy \
  --exclude-module aiofiles --exclude-module fsspec \
  --exclude-module dask --exclude-module distributed \
  --exclude-module xarray --exclude-module numba \
  --exclude-module llvmlite --exclude-module h5py \
  --exclude-module tables --exclude-module zarr \
  --exclude-module flask --exclude-module fastapi \
  --exclude-module uvicorn --exclude-module starlette \
  --exclude-module websockets \
  --copy-metadata onnxruntime \
  --copy-metadata numpy \
  --copy-metadata aiohttp \
  "$ROOT/server.py" \
  --distpath "$DIST" --workpath /tmp/ddz_build --specpath /tmp/ddz_spec 2>&1 | tail -8

BUILD_DIR="$DIST/斗地主"
if [ ! -d "$BUILD_DIR" ]; then
  echo "❌ PyInstaller 未产出 $BUILD_DIR"
  exit 1
fi

echo "==> 组装 .app"
APP_PATH="$DIST/$APP"
rm -rf "$APP_PATH"
mkdir -p "$APP_PATH/Contents/MacOS" "$APP_PATH/Contents/Resources"

# PyInstaller 产物整体放进 Resources
mv "$BUILD_DIR" "$APP_PATH/Contents/Resources/_internal"

cat > "$APP_PATH/Contents/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key>            <string>斗地主</string>
  <key>CFBundleDisplayName</key>     <string>斗地主 · DouZero AI</string>
  <key>CFBundleIdentifier</key>      <string>com.local.doudizhu-py</string>
  <key>CFBundleVersion</key>         <string>1.0.0</string>
  <key>CFBundleExecutable</key>      <string>launcher</string>
  <key>CFBundlePackageType</key>     <string>APPL</string>
  <key>LSUIElement</key>             <false/>
  <key>NSHighResolutionCapable</key> <true/>
</dict>
</plist>
PLIST

# 启动器: 打开终端跑服务, 关窗口即退出
cat > "$APP_PATH/Contents/MacOS/launcher" <<'LAUNCH'
#!/usr/bin/env bash
RES="$(cd "$(dirname "$0")/../Resources/_internal" && pwd)"
export DDZ_REPLAY_DIR="$HOME/Library/Application Support/doudizhu/replays"
mkdir -p "$DDZ_REPLAY_DIR"
PORT="${DDZ_PORT:-8000}"
clear
cat <<'BANNER'
  ┌──────────────────────────────────────┐
  │       斗 地 主 · DouZero AI           │
  │                                      │
  │   本窗口就是服务端, 不要关              │
  │   把下面打印的地址发给同事开浏览器玩    │
  └──────────────────────────────────────┘
BANNER
echo ""
exec "$RES/斗地主" "$PORT"
LAUNCH
chmod +x "$APP_PATH/Contents/MacOS/launcher"

# ad-hoc 签名
codesign --force --deep --sign - "$APP_PATH" 2>/dev/null || true

# zip 便于分发
ZIPNAME="doudizhu-py.app.zip"
cd "$DIST"
rm -f "$ZIPNAME"
ditto -c -k --sequesterRsrc --keepParent "$APP" "$ZIPNAME"
cd "$ROOT"

echo ""
echo "✅ 打包完成:"
echo "   $APP_PATH        ($(du -sh "$APP_PATH" | cut -f1))  <- 双击直接用"
echo "   $DIST/$ZIPNAME   ($(du -sh "$DIST/$ZIPNAME" | cut -f1))  <- 拷给同事"
