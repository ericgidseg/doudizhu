#!/usr/bin/env bash
# 打成可分发的 macOS .app —— 双击运行, 无需装 Node
# 产物: dist/斗地主.app  (含 node arm64 二进制 + bundle + public)
# 同事拿到 .app → 拖到应用程序/桌面 → 双击 → 弹出的终端窗口不要关, 浏览器打开打印的地址
set -e
cd "$(dirname "$0")"
APP="${1:-斗地主.app}"
DIST="dist"
NODE_BIN="$(command -v node)"
echo "使用 Node: $NODE_BIN ($(node -v))"

# ---------- 1. 单文件 bundle (内联本地 require) ----------
cat > make-bundle.js <<'EOF'
const fs = require('fs');
const LOCAL = ['cards.js', 'ai.js', 'ws.js'];
function inline(code) {
  return code.replace(/require\('\.\/(cards|ai|ws)\.js'\)/g, "__req('$1.js')");
}
function wrap(n) {
  return `__modules['${n}']=function(module,require){\n${inline(fs.readFileSync(n,'utf8'))}\n};`;
}
let out = `'use strict';\nconst __modules={};\nfunction __req(n){ if(!__modules[n]) return require(n); const m={exports:{}};__modules[n](m,__req);return m.exports;}\n`;
for (const f of LOCAL) out += wrap(f) + '\n';
out += inline(fs.readFileSync('server.js', 'utf8'));
fs.writeFileSync('bundle.js', out);
EOF
node make-bundle.js
node --check bundle.js

# ---------- 2. 组装 .app ----------
rm -rf "$DIST/$APP"
mkdir -p "$DIST/$APP/Contents/MacOS" "$DIST/$APP/Contents/Resources"

# Info.plist
cat > "$DIST/$APP/Contents/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key>            <string>斗地主</string>
  <key>CFBundleDisplayName</key>     <string>斗地主 · 内网对战</string>
  <key>CFBundleIdentifier</key>      <string>com.local.doudizhu</string>
  <key>CFBundleVersion</key>         <string>1.0.0</string>
  <key>CFBundleExecutable</key>      <string>launcher</string>
  <key>CFBundlePackageType</key>     <string>APPL</string>
  <key>LSUIElement</key>             <false/>
  <key>NSHighResolutionCapable</key> <true/>
</dict>
</plist>
PLIST

# node 运行时 + 依赖库
REAL_NODE="$(realpath "$NODE_BIN")"
cp -L "$REAL_NODE" "$DIST/$APP/Contents/Resources/node"
chmod +x "$DIST/$APP/Contents/Resources/node"
LIBDIR="$(dirname "$REAL_NODE")/../lib"
for lib in "$LIBDIR"/libnode*.dylib; do
  [ -e "$lib" ] && cp -L "$lib" "$DIST/$APP/Contents/Resources/" && echo "  带库: $(basename "$lib")"
done

# 代码 + 静态资源
cp bundle.js "$DIST/$APP/Contents/Resources/"
cp -r public "$DIST/$APP/Contents/Resources/public"

# ---------- 3. 启动器 (打开终端跑服务, 关窗口即退出) ----------
cat > "$DIST/$APP/Contents/MacOS/launcher" <<'LAUNCH'
#!/usr/bin/env bash
# .app 双击入口: 切到 Resources 起 node 服务, replays 写到 ~/Library/Application Support
RES="$(cd "$(dirname "$0")/../Resources" && pwd)"
export DYLD_LIBRARY_PATH="$RES:${DYLD_LIBRARY_PATH}"
export DDZ_REPLAY_DIR="$HOME/Library/Application Support/doudizhu/replays"
mkdir -p "$DDZ_REPLAY_DIR"
PORT="${DDZ_PORT:-8000}"
clear
cat <<'BANNER'
  ┌──────────────────────────────────────┐
  │       斗 地 主 · 内 网 对 战            │
  │                                      │
  │   本窗口就是服务端, 不要关              │
  │   把下面打印的地址发给同事开浏览器玩    │
  └──────────────────────────────────────┘
BANNER
echo ""
exec "$RES/node" "$RES/bundle.js" "$PORT"
LAUNCH
chmod +x "$DIST/$APP/Contents/MacOS/launcher"

# ---------- 4. ad-hoc 签名 ----------
# macOS 对完全未签名的二进制会直接 SIGKILL ("代码签名无效"对话框,
# 提示联系开发者重新安装). ad-hoc 签名不解决公证, 但让本机直接双击可用.
codesign --force --deep --sign - "$DIST/$APP"
codesign --verify --strict "$DIST/$APP"

# ---------- 5. 顺手产出 zip 便于分发 ----------
# zip 文件名用 ASCII, 避免 GitHub Release 清洗中文名
ZIPNAME="doudizhu.app.zip"
cd "$DIST"
rm -f "$ZIPNAME"
ditto -c -k --sequesterRsrc --keepParent "$APP" "$ZIPNAME"
cd ..

rm -f bundle.js make-bundle.js
echo ""
echo "✅ 打包完成:"
echo "   $DIST/$APP        ($(du -sh "$DIST/$APP" | cut -f1))  <- 双击直接用"
echo "   $DIST/$ZIPNAME    ($(du -sh "$DIST/$ZIPNAME" | cut -f1))  <- 拷给同事解压即用"
echo ""
echo "同事首次双击可能被 Gatekeeper 拦 (未签名), 让他:"
echo "   右键 .app → 打开 → 打开   (一次以后即可)"
echo "或对 itch.io 分发级: 在 '系统设置 → 隐私与安全性' 里点 '仍要打开'"
