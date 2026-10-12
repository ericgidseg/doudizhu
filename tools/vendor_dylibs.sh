#!/usr/bin/env bash
# 递归收集可执行文件的非系统动态库依赖, 拷入 DEST 并把加载路径改写为
# @loader_path, 最后 ad-hoc 重签.
# 背景: Homebrew 的 node 除 libnode 外还直接链接 simdjson/abseil/llhttp
# 等十几库, 只打包 libnode 时, 没装相同版本 Homebrew 库的机器上 dyld
# 会在启动瞬间杀进程 ("Library not loaded ... libsimdjson.33.dylib").
# 用法: vendor_dylibs.sh <node二进制> <node的lib目录(解析@rpath用)> <输出目录>
set -euo pipefail

NODE_SRC="$1"
NODE_LIBDIR="$2"
DEST="$3"
mkdir -p "$DEST"

# 系统库每台 macOS 都自带, 不打包
is_system() {
  case "$1" in
    /System/*|/usr/lib/*) return 0 ;;
    *) return 1 ;;
  esac
}

# 已收集的库 (V_NAMES/V_SRCS 平行数组, 线性查找, 兼容 macOS 自带 bash 3.2)
V_NAMES=()
V_SRCS=()
v_find_src() {  # $1=basename -> 输出原始路径, 没找到返回 1
  [ ${#V_NAMES[@]} -gt 0 ] || return 1
  local i
  for i in "${!V_NAMES[@]}"; do
    [ "${V_NAMES[$i]}" = "$1" ] && { printf '%s' "${V_SRCS[$i]}"; return 0; }
  done
  return 1
}

# 待处理队列: Q_FILES 是 DEST 里的副本, Q_SRCS 是构建机上的原始路径
Q_FILES=()
Q_SRCS=()

resolve_dep() {  # $1=引用者副本 $2=引用者原始目录 $3=依赖串 -> 输出原始真实路径
  local ref="$1" orig="$2" dep="$3" name rp cand
  name="$(basename "$dep")"
  case "$dep" in
    /*)
      [ -e "$dep" ] && { printf '%s' "$dep"; return 0; }
      return 1 ;;
    @rpath/*|@executable_path/*|@loader_path/*)
      for cand in "$NODE_LIBDIR/$name" "$orig/$name"; do
        [ -e "$cand" ] && { printf '%s' "$cand"; return 0; }
      done
      while IFS= read -r rp; do
        [ -e "$rp/$name" ] && { printf '%s' "$rp/$name"; return 0; }
      done < <(otool -l "$ref" | awk '/LC_RPATH/{f=1;next} f&&$1=="path"{print $2;f=0}')
      return 1 ;;
    *) return 1 ;;
  esac
}

cp -L "$NODE_SRC" "$DEST/node"
chmod +x "$DEST/node"
Q_FILES+=("$DEST/node")
Q_SRCS+=("$NODE_SRC")

while [ ${#Q_FILES[@]} -gt 0 ]; do
  CUR="${Q_FILES[0]}"
  SRCDIR="$(dirname "${Q_SRCS[0]}")"
  Q_FILES=("${Q_FILES[@]:1}")
  Q_SRCS=("${Q_SRCS[@]:1}")
  DEPS="$(otool -L "$CUR" | tail -n +2 | awk '{print $1}')"
  for DEP in $DEPS; do
    is_system "$DEP" && continue
    if ! SRC="$(resolve_dep "$CUR" "$SRCDIR" "$DEP")"; then
      echo "  !! 无法解析依赖: $CUR -> $DEP" >&2
      exit 1
    fi
    # 同一物理文件可能经 Cellar 与 /opt/homebrew/opt 两条路径出现,
    # 按真实路径去重, 避免误报同名不同源
    SRC="$(realpath "$SRC")"
    NAME="$(basename "$SRC")"
    if V_SRC="$(v_find_src "$NAME")"; then
      if [ "$V_SRC" != "$SRC" ]; then
        echo "  !! 同名不同源的库: $NAME ($V_SRC 与 $SRC)" >&2
        exit 1
      fi
    else
      cp -L "$SRC" "$DEST/$NAME"
      V_NAMES+=("$NAME")
      V_SRCS+=("$SRC")
      Q_FILES+=("$DEST/$NAME")
      Q_SRCS+=("$SRC")
      echo "  带库: $NAME"
    fi
    install_name_tool -change "$DEP" "@loader_path/$NAME" "$CUR"
  done
done

# 库自身的 install id 也统一指向 @loader_path, 避免按旧路径再解析
for NAME in "${V_NAMES[@]}"; do
  install_name_tool -id "@loader_path/$NAME" "$DEST/$NAME" || true
done

# install_name_tool 改写会使原签名失效, 全部 ad-hoc 重签
codesign --force --sign - "$DEST/node"
for NAME in "${V_NAMES[@]}"; do
  codesign --force --sign - "$DEST/$NAME"
done
