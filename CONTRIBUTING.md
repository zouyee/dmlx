# Contributing to MLX-Zig

We want to make contributing to this project as easy and transparent as
possible.

## Pull Requests

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure `zig build test` passes with no failures or leaks.
5. Run `zig fmt src/` to ensure consistent formatting.

## Development Setup

```bash
# Clone
git clone https://github.com/zouyee/mlx-zig.git
cd mlx-zig

# Build
zig build

# Test
zig build test

# Run example
./zig-out/bin/example

# Format
zig fmt src/
```

## Code Style

- Follow Zig standard library conventions.
- Use `snake_case` for functions and variables, `PascalCase` for types.
- Public API functions take an `EagerContext` or `Context` as the first parameter.
- All allocations must be properly freed — `zig build test` uses
  `std.testing.allocator` which detects leaks.
- Prefer `comptime` dispatch over runtime branching for dtype-specific code.

## Adding a New Operation

1. Add the implementation in the appropriate file under `src/ops/`.
2. Export it through `src/root.zig` if it's a new module.
3. Add tests in `src/tests.zig` or the relevant test file under `src/tests/`.
4. Update `MIGRATION.md` if it fills a gap vs MLX C++.

## Testing

Tests are organized as follows:

| File | Purpose |
|------|---------|
| `src/tests.zig` | Main test suite (167 tests) |
| `src/tests/autograd_tests.zig` | Autograd VJP/JVP tests |
| `src/tests/fft_tests.zig` | FFT tests |
| `src/tests/linalg_tests.zig` | Linear algebra tests |
| `src/tests/random_tests.zig` | Random distribution tests |
| `src/tests/einsum_tests.zig` | Einsum equivalence tests |
| `src/tests/scheduler_tests.zig` | Scheduler tests |
| `src/tests/device_tests.zig` | Device/stream tests |
| `src/tests/allocator_tests.zig` | Memory management tests |
| `src/tests/io_tests.zig` | I/O format tests |
| `src/tests/compile_tests.zig` | Compilation tests |
| `src/tests/aligned_tests.zig` | Aligned property tests |
| `src/tests/metal_test.zig` | Metal GPU tests |
| `src/tests/property_*.zig` | Property-based tests |

## Issues

Use GitHub issues to report bugs. Please include:
- Zig version (`zig version`)
- OS and architecture
- Minimal reproduction code
- Expected vs actual behavior

## License

By contributing to MLX-Zig, you agree that your contributions will be licensed
under the MIT License in the root directory of this source tree.
