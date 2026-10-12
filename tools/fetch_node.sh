#!/usr/bin/env bash
# 获取打包用的官方静态 Node 运行时 (stdout 输出 node 路径).
#
# 为什么不用本机 node: Homebrew 版除 libnode 外还链接 simdjson/abseil
# 等近百个动态库, 打包需全部收集且他机易因缺库启动即崩; 官方二进制
# 只依赖系统框架, 自包含且 strip 调试符号后小 ~20MB.
#
# 版本可用 NODE_DIST_VERSION=v22.23.3 覆盖; 缓存命中且可用时直接复用.
set -euo pipefail
cd "$(dirname "$0")/.."

VER="${NODE_DIST_VERSION:-v22.23.3}"
case "$(uname -m)" in
  arm64)  DIST_ARCH=arm64 ;;
  x86_64) DIST_ARCH=x64 ;;
  *) echo "不支持的架构: $(uname -m)" >&2; exit 1 ;;
esac

DIST="node-$VER-darwin-$DIST_ARCH"
CACHE=".cache/node/$DIST"
BIN="$CACHE/bin/node"

if [ -x "$BIN" ] && "$BIN" -v >/dev/null 2>&1; then
  printf '%s\n' "$(pwd)/$BIN"
  exit 0
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
BASE="https://nodejs.org/dist/$VER"

echo "下载官方 Node $VER ($DIST_ARCH)..." >&2
curl -fsSL --retry 3 -o "$TMP/$DIST.tar.gz" "$BASE/$DIST.tar.gz"
curl -fsSL --retry 3 -o "$TMP/SHASUMS256.txt" "$BASE/SHASUMS256.txt"
(cd "$TMP" && grep " $DIST.tar.gz\$" SHASUMS256.txt | shasum -a 256 -c -) >&2

tar -xzf "$TMP/$DIST.tar.gz" -C "$TMP"
# strip 去掉调试符号省 ~20MB; 会破坏官方签名, arm64 强制要求有效
# 签名, 因此必须 ad-hoc 重签后再使用.
strip "$TMP/$DIST/bin/node" 2>/dev/null || true
codesign --force --sign - "$TMP/$DIST/bin/node"

mkdir -p "$CACHE/bin"
mv "$TMP/$DIST/bin/node" "$BIN"
chmod +x "$BIN"
"$BIN" -v >&2
printf '%s\n' "$(pwd)/$BIN"
