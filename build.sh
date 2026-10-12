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
# node 若为动态链接, 递归收集全部依赖库并改写为 @loader_path
# (Homebrew node 链接了 simdjson/abseil 等十几库, 只带 libnode 会在
# 没装相同版本 Homebrew 库的机器上启动即崩)
LIBDIR="$(dirname "$REAL_NODE")/../lib"
REAL_NODE="$(realpath "$NODE_BIN")"
LIBDIR="$(dirname "$REAL_NODE")/../lib"
tools/vendor_dylibs.sh "$REAL_NODE" "$LIBDIR" .pkg
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
clear
echo "  ┌─────────────────────────────────────┐"
echo "  │      斗 地 主 · 内 网 对 战           │"
echo "  │   把这个窗口开着, 同事就能加入        │"
echo "  └─────────────────────────────────────┘"
echo ""
./node bundle.js "$PORT" &
NODE_PID=$!
printf '启动中'
while ! curl -s -o /dev/null --max-time 1 "http://localhost:$PORT/" 2>/dev/null; do
  kill -0 "$NODE_PID" 2>/dev/null || { echo; echo "服务启动失败"; exit 1; }
  printf '.'; sleep 1
done
echo
wait "$NODE_PID"
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
