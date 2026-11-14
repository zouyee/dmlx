# Homebrew formula for dmlx (self-contained binary release).
#
# This file belongs in a Homebrew tap repository, e.g.
#   github.com/zouyee/homebrew-tap/Formula/dmlx.rb
# Users then install with:
#   brew install zouyee/tap/dmlx
#
# The release tarball already bundles libmlxc/libmlx/mlx.metallib, so there
# is no mlx-c dependency. Update `url` and `sha256` per release (the release
# workflow emits dmlx-<tag>-macos-arm64.tar.gz{,.sha256}).
class Dmlx < Formula
  desc "Zig inference runtime for Apple Silicon with speculative decoding"
  homepage "https://github.com/zouyee/dmlx"
  url "https://github.com/zouyee/dmlx/releases/download/v0.0.8/dmlx-v0.0.8-macos-arm64.tar.gz"
  sha256 "REPLACE_WITH_SHA256_FROM_RELEASE"
  license "MIT"

  depends_on arch: :arm64
  depends_on :macos

  def install
    libexec.install "bin/dmlx"
    libexec.install "lib"
    # Wrapper keeps the real binary path stable so @executable_path/../lib
    # resolves to the bundled dylibs inside libexec.
    (bin/"dmlx").write <<~EOS
      #!/bin/bash
      exec "#{libexec}/bin/dmlx" "$@"
    EOS
  end

  def caveats
    <<~EOS
      dmlx is ad-hoc signed. If Gatekeeper blocks it, run:
        xattr -dr com.apple.quarantine #{libexec}
      Models are not included; point --model at a local checkpoint.
    EOS
  end

  test do
    assert_match "dmlx", shell_output("#{bin}/dmlx version")
  end
end
