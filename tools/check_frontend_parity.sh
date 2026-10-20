#!/usr/bin/env bash
# 校验两份前端 (public/client.js 与 python/public/client.js) 的牌面渲染
# 逻辑一致.
#
# 两份前端有意保留三处差异: WebSocket 路径 (Node 版直接连, Python 版带
# /ws), turnBanner 处理, 回合倒计时. 所以不能整体 diff.
#
# 但牌面排序/编码这类"漏改一次就静默变蠢"的共享逻辑, 必须两边相同.
# 历史上已因漏改出过三次事故: 编码错位 / 牌序漏在 Python 版 / 版本落后.
# 用法: ./tools/check_frontend_parity.sh   (CI 里每次构建前跑)
set -euo pipefail
cd "$(dirname "$0")/.."

NODE=public/client.js
PY=python/public/client.js

# 共享逻辑块的起止标记. 块之间允许排版差异, 抽取后统一去掉空白再比.
# 新增共享逻辑时, 在这里加一行 block name + 起止正则.
extract() {
  # $1=file  $2=start-pattern  $3=end-pattern
  sed -n "/$2/,/$3/p" "$1" | sed 's/[[:space:]]//g' | grep -v '^$' || true
}

fail=0

check_block() {
  local name="$1" start="$2" end="$3"
  local a b
  a=$(extract "$NODE" "$start" "$end")
  b=$(extract "$PY" "$start" "$end")
  if [ -z "$a" ]; then
    echo "❌ $name: $NODE 里没找到 ($start)"
    return 1
  fi
  if [ "$a" = "$b" ]; then
    echo "✅ $name"
    return 0
  fi
  echo "❌ $name: 两份前端不一致"
  diff <(echo "$a") <(echo "$b") | head -20 || true
  return 1
}

# check_block 用 return code 报告失败 (函数内改 fail 会因 $( ) 子 shell 丢失)
check_block "出牌区排序" '出牌区牌序' 'const sortPlayed =' || fail=1


# 手牌/出牌区的渲染入口也必须同源: renderPlayed 的那一行.
# 不带 -n: 两份文件行数本就不同 (Python 版多倒计时), 行号参与比较会误报.
a=$(grep 'sortPlayed(lp.cards' "$NODE" | sed 's/[[:space:]]//g')
b=$(grep 'sortPlayed(lp.cards' "$PY" | sed 's/[[:space:]]//g')
if [ "$a" = "$b" ]; then
  echo "✅ renderPlayed 调用点"
else
  echo "❌ renderPlayed 调用点不一致: [$a] vs [$b]"
  fail=1
fi

if [ "$fail" -ne 0 ]; then
  echo ""
  echo "两份前端必须同步修改. 若新增差异是有意的, 请同步更新本脚本."
  exit 1
fi
echo ""
echo "✅ 前端共享逻辑一致"
