# AGENTS.md

Guidance for AI agents working with or inside this repository.

## If the user wants to RUN dmlx (not develop it)

Do not build from source. Use the release install:

```bash
curl -fsSL https://raw.githubusercontent.com/zouyee/dmlx/main/scripts/install.sh | bash
dmlx doctor
dmlx pull qwen3.8-27b-4bit && dmlx pull qwen3.8-27b-mtp-4bit
dmlx serve --model qwen3.8-27b-4bit --speculative-mtp 7 --port 8080
```

See [llms.txt](llms.txt) for the full run/integrate reference, including the
DeepSeek V4 native path and its memory constraints.

## If the user wants to DEVELOP dmlx

- Build: `zig build -Doptimize=ReleaseFast` (Zig 0.16+, `mlx-c` via pkg-config
  or `MLX_C_PREFIX`). Binary: `zig-out/bin/dmlx`.
- Test: `make test`. Metal kernel changes also need
  `bash scripts/run_kernel_tests.sh` and `bash scripts/run_mla_attention_test.sh`.
- Benchmarks: `make benchmark` (requires local models). Attach raw artifacts
  when reporting numbers; see [README.md](README.md#benchmarks) for the
  measurement protocol.
- Source layout and per-file responsibilities: [README.md](README.md#repository-map).
- Qwen3.8 MTP engineering history and remaining limits: [HANDOFF.md](HANDOFF.md).
