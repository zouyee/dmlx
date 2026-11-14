#!/bin/bash
# One-command install for dmlx on Apple Silicon macOS.
#
#   curl -fsSL https://raw.githubusercontent.com/zouyee/dmlx/main/scripts/install.sh | bash
#
# Installs the latest release into ~/.local/share/dmlx and symlinks the CLI
# into ~/.local/bin/dmlx. No sudo, no Zig, no other dependencies: the release
# tarball bundles libmlxc, libmlx, and mlx.metallib.
set -euo pipefail

REPO="zouyee/dmlx"
PREFIX="${DMLX_INSTALL_PREFIX:-$HOME/.local}"
DEST="$PREFIX/share/dmlx"
BIN_DIR="$PREFIX/bin"

fail() { echo "install.sh: $*" >&2; exit 1; }

[ "$(uname -s)" = "Darwin" ] || fail "dmlx currently supports macOS only"
[ "$(uname -m)" = "arm64" ] || fail "dmlx requires Apple Silicon (arm64)"

command -v curl >/dev/null || fail "curl is required"

echo "==> Resolving latest release"
ASSET_URL=$(curl -fsSL "https://api.github.com/repos/$REPO/releases/latest" \
  | grep '"browser_download_url"' \
  | grep 'macos-arm64\.tar\.gz"' \
  | cut -d'"' -f4)
[ -n "$ASSET_URL" ] || fail "no macos-arm64 tarball found in latest release"
SHA_URL="$ASSET_URL.sha256"
NAME=$(basename "$ASSET_URL" .tar.gz)
echo "    $NAME"

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

echo "==> Downloading"
curl -fSL --progress-bar "$ASSET_URL" -o "$TMP/$NAME.tar.gz"
if curl -fsSL "$SHA_URL" -o "$TMP/$NAME.tar.gz.sha256" 2>/dev/null; then
  echo "==> Verifying sha256"
  (cd "$TMP" && shasum -a 256 -c "$NAME.tar.gz.sha256" >/dev/null) \
    || fail "sha256 mismatch; aborting"
else
  echo "    (no .sha256 asset; skipping verification)"
fi

echo "==> Installing to $DEST"
mkdir -p "$DEST" "$BIN_DIR"
tar -xzf "$TMP/$NAME.tar.gz" -C "$DEST" --strip-components 1

# The binary is ad-hoc signed; clear quarantine so Gatekeeper does not block it.
xattr -dr com.apple.quarantine "$DEST" 2>/dev/null || true

ln -sf "$DEST/bin/dmlx" "$BIN_DIR/dmlx"

echo "==> Installed to $BIN_DIR/dmlx"
"$BIN_DIR/dmlx" version || true
case ":$PATH:" in
  *":$BIN_DIR:"*) ;;
  *) echo ""
     echo "Add $BIN_DIR to your PATH, e.g.:"
     echo "  echo 'export PATH=\"$BIN_DIR:\$PATH\"' >> ~/.zshrc && source ~/.zshrc" ;;
esac

cat <<EOF

Next steps:
  dmlx doctor                          # check hardware, Metal, models
  dmlx pull qwen3.8-27b-4bit           # download the flagship model
  dmlx pull qwen3.8-27b-mtp-4bit       # MTP drafter (speculative decode)
  dmlx serve --model qwen3.8-27b-4bit --speculative-mtp 7 --port 8080
EOF
