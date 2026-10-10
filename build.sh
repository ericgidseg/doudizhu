#!/usr/bin/env bash
# 打成 "双击即可运行" 的单文件 .command (macOS) —— 内含 node 二进制 + 代码 + 静态资源
# 用法: ./build.sh [输出名]   默认 ./斗地主.command
set -e
cd "$(dirname "$0")"
OUT="${1:-斗地主.command}"
NODE_BIN="$(command -v node)"
echo "使用 Node: $NODE_BIN ($(node -v))"

# 1. 生成单文件 bundle (本地 require 内联)
cat > make-bundle.js <<'EOF'
const fs = require('fs');
const LOCAL = ['cards.js', 'ai.js', 'ws.js'];
// 把所有 require('./x.js') 统一替换为 __req('x.js')
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

# 2. 打包 payload: node 二进制 + 动态库 + bundle + public/
rm -rf .pkg && mkdir -p .pkg
REAL_NODE="$(realpath "$NODE_BIN")"
cp -L "$NODE_BIN" .pkg/node
chmod +x .pkg/node
# node 若为动态链接, 带上 libnode.dylib (macOS Homebrew 编译版常见)
LIBDIR="$(dirname "$REAL_NODE")/../lib"
for lib in "$LIBDIR"/libnode*.dylib; do
  [ -e "$lib" ] && cp -L "$lib" .pkg/ && echo "  打包库: $(basename "$lib")"
done
cp bundle.js .pkg/
cp -r public .pkg/
PAYLOAD=$(mktemp)
tar -czf "$PAYLOAD" -C .pkg .
rm -rf .pkg

# 3. 自解压启动器
cat > "$OUT" <<'LAUNCHER'
#!/usr/bin/env bash
# 斗地主 · 内网版 —— 双击运行, 无需安装 Node
PORT="${1:-8000}"
DIR="$(mktemp -d /tmp/doudizhu.XXXXXX)"
ARCHIVE_LINE=$(awk '/^__PAYLOAD_BELOW__$/ {print NR + 1; exit}' "$0")
tail -n +$ARCHIVE_LINE "$0" | tar -xzf - -C "$DIR" 2>/dev/null || {
  tail -n +$ARCHIVE_LINE "$0" | tar -xz -C "$DIR"
}
cd "$DIR"
export DYLD_LIBRARY_PATH="$DIR:$DIR/../lib:${DYLD_LIBRARY_PATH}"
clear
echo "  ┌─────────────────────────────────────┐"
echo "  │      斗 地 主 · 内 网 对 战           │"
echo "  │   把这个窗口开着, 同事就能加入        │"
echo "  └─────────────────────────────────────┘"
echo ""
./node bundle.js "$PORT"
# 退出时自动清理
rm -rf "$DIR"
exit
__PAYLOAD_BELOW__
LAUNCHER
cat "$PAYLOAD" >> "$OUT"
rm -f "$PAYLOAD" bundle.js make-bundle.js
chmod +x "$OUT"

echo ""
echo "✅ 打包完成: ./$OUT  ($(du -h "$OUT" | cut -f1))"
echo "   双击即可运行, 或终端: ./$OUT [端口]"
echo "   拷给同事 → 双击 → 把打印的地址发出去"
