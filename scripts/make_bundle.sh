#!/usr/bin/env bash
# Bundle dmlx into a self-contained tarball that does not require Homebrew,
# Zig, or any user-installed dylib at runtime.
#
# Copies zig-out/bin/dmlx plus the mlx-c / mlx dynamic libraries and
# mlx.metallib into dist/dmlx/{bin,lib}, rewrites install names to @rpath,
# ad-hoc codesigns the result, and produces
#   dist/dmlx-<version>-macos-arm64.tar.gz (+ .sha256)
#
# Layout produced:
#   dmlx/bin/dmlx            (rpath: @executable_path/../lib)
#   dmlx/lib/libmlxc.dylib   (id: @rpath/libmlxc.dylib)
#   dmlx/lib/libmlx.dylib    (id: @rpath/libmlx.dylib)
#   dmlx/lib/mlx.metallib    (found by libmlx via dladdr-relative lookup)
#
# Usage: scripts/make_bundle.sh [path-to-dmlx-binary]

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DMLX_BIN="${1:-$PROJECT_DIR/zig-out/bin/dmlx}"
DIST_DIR="$PROJECT_DIR/dist"
BUNDLE_ROOT="$DIST_DIR/dmlx"

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
    echo "error: bundling is only supported on macOS arm64" >&2
    exit 1
fi
if [[ ! -x "$DMLX_BIN" ]]; then
    echo "error: dmlx binary not found at $DMLX_BIN (run 'make' first)" >&2
    exit 1
fi

# Locate the mlx-c dylib actually linked into the binary, resolving any
# brew-style /opt symlink to the real Cellar file.
linked_mlxc="$(otool -L "$DMLX_BIN" | awk '/libmlxc\.dylib/ {print $1; exit}')"
if [[ -z "$linked_mlxc" ]]; then
    echo "error: $DMLX_BIN does not link libmlxc.dylib" >&2
    exit 1
fi
MLXC_LIB="$(realpath "$linked_mlxc")"
MLX_C_PREFIX="$(dirname "$(dirname "$MLXC_LIB")")"

# libmlxc depends on libmlx from the separate 'mlx' formula.
MLX_LIB="$(realpath "$(otool -L "$MLXC_LIB" | awk '/libmlx\.dylib/ {print $1; exit}')")"
MLX_PREFIX="$(dirname "$(dirname "$MLX_LIB")")"
METALLIB="$MLX_PREFIX/lib/mlx.metallib"

for f in "$MLXC_LIB" "$MLX_LIB" "$METALLIB"; do
    if [[ ! -f "$f" ]]; then
        echo "error: required file missing: $f" >&2
        exit 1
    fi
done

# Version: git tag/describe, else 'dev'.
VERSION="$(git -C "$PROJECT_DIR" describe --tags --always --dirty 2>/dev/null || echo dev)"

rm -rf "$BUNDLE_ROOT"
mkdir -p "$BUNDLE_ROOT/bin" "$BUNDLE_ROOT/lib"
cp "$DMLX_BIN" "$BUNDLE_ROOT/bin/dmlx"
cp "$MLXC_LIB" "$BUNDLE_ROOT/lib/libmlxc.dylib"
cp "$MLX_LIB" "$BUNDLE_ROOT/lib/libmlx.dylib"
cp "$METALLIB" "$BUNDLE_ROOT/lib/mlx.metallib"
chmod +w "$BUNDLE_ROOT/bin/dmlx" "$BUNDLE_ROOT/lib/"*.dylib

# Rewrite install names.
install_name_tool -id "@rpath/libmlxc.dylib" "$BUNDLE_ROOT/lib/libmlxc.dylib"
install_name_tool -id "@rpath/libmlx.dylib" "$BUNDLE_ROOT/lib/libmlx.dylib"
install_name_tool -change "$linked_mlxc" "@rpath/libmlxc.dylib" "$BUNDLE_ROOT/bin/dmlx"
install_name_tool -change \
    "$(otool -L "$MLXC_LIB" | awk '/libmlx\.dylib/ {print $1; exit}')" \
    "@rpath/libmlx.dylib" "$BUNDLE_ROOT/lib/libmlxc.dylib"
install_name_tool -add_rpath "@executable_path/../lib" "$BUNDLE_ROOT/bin/dmlx"
install_name_tool -add_rpath "@loader_path" "$BUNDLE_ROOT/lib/libmlxc.dylib"

# Drop any absolute rpaths inherited from the build machine (e.g. the brew
# prefix the linker recorded). Without this, dyld would silently prefer a
# Homebrew-installed libmlxc over the bundled one.
for f in "$BUNDLE_ROOT/bin/dmlx" "$BUNDLE_ROOT/lib/"*.dylib; do
    otool -l "$f" | awk '/LC_RPATH/{getline; getline; if ($2 ~ /^\//) print $2}' | while read -r rp; do
        install_name_tool -delete_rpath "$rp" "$f"
    done
done

# Modifying the binaries invalidates any previous signature; ad-hoc sign so
# the loader accepts them (Developer ID signing + notarization is M4).
codesign --force --sign - "$BUNDLE_ROOT/bin/dmlx" "$BUNDLE_ROOT/lib/"*.dylib

# Gate: no absolute Homebrew/Cellar path may remain in any bundled binary.
fail=0
for f in "$BUNDLE_ROOT/bin/dmlx" "$BUNDLE_ROOT/lib/"*.dylib; do
    if otool -L "$f" | grep -E '/opt/homebrew|/Cellar'; then
        echo "error: $f still references an absolute Homebrew path" >&2
        fail=1
    fi
done
[[ "$fail" == "0" ]] || exit 1

# Smoke: run the bundled binary; it must load without brew dylibs.
"$BUNDLE_ROOT/bin/dmlx" version >/dev/null

TARBALL="$DIST_DIR/dmlx-${VERSION}-macos-arm64.tar.gz"
rm -f "$TARBALL" "$TARBALL.sha256"
tar -czf "$TARBALL" -C "$DIST_DIR" dmlx
shasum -a 256 "$TARBALL" | awk '{print $1"  "FILENAME}' FILENAME="$(basename "$TARBALL")" > "$TARBALL.sha256"

echo "bundle OK: $BUNDLE_ROOT"
echo "tarball:   $TARBALL"
echo "checksum:  $TARBALL.sha256"
