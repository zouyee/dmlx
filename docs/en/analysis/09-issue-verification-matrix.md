# Chapter 9: Issue Verification Matrix

## 9.1 Cross-Verification with Project Self-Audit

`docs/deep-analysis.md` is the self-audit document from v0.3.0, and `ROADMAP.md` claims all issues have been fixed. This analysis verifies each claim:

| Original Issue | Original Severity | Project Claim | Actual Status | Deviation |
|--------|---------|---------|------|------|
| Systemic memory leaks | P0 | ✅ Fixed | **Partially fixed** | `ScopedArrayArena` introduced, but `nn.zig` CPU path bypasses Arena |
| Error message loss | P0 | ✅ Fixed | **Fixed** | `mlxErrorHandler` correctly captures C++ exception text |
| NN/Activation bypass GPU | P0 | ✅ Fixed | **Partially fixed** | `activations.zig` fully GPU-ized; `nn.zig` still has 34 `dataSliceMut` |
| Sampling insertion sort | P2 | Not mentioned | **Not fixed** | 4 calls still present |
| `dataSliceMut` @constCast | P1 | ✅ Fixed | **Not fixed** | 10 `@constCast` + `nn.zig` 34 `dataSliceMut` across repo |
| Hardcoded Homebrew | P1 | ✅ Fixed | **Fixed** | Four-level detection implemented |
| EagerContext stream leak | P1 | Not mentioned | **Not fixed** | Still no `deinit` |
| Attention mask ignored | P1 | Not mentioned | **Pending verification** | `nn.zig` TransformerEncoderLayer needs confirmation |
| allocator parameter misleading | P2 | Not mentioned | **Not fixed** | `array.zig` 3 instances of `_ = allocator` |
| ops.zig vs ops/ duplication | P2 | Not mentioned | **Not fixed** | Two APIs coexist |
| zig-regex pointing to main | P1 | ✅ Fixed | **Fixed** | Now pinned to fixed hash |
| NN layers no tests | P1 | ✅ Fixed | **Partially fixed** | No `nn_tests`, `numerical_equivalence_test` covers some |
| Autograd no tests | P1 | Not mentioned | **Not fixed** | No `grad_tests` |
| Missing golden test | P1 | ✅ Fixed | **Partially fixed** | Golden files exist but use random weights |

## 9.2 Fix Completion Statistics

```
P0 Issues (3)
├── Systemic memory leaks     Partially fixed  ⚠️
├── Error message loss        Fixed            ✅
└── NN/Activation GPU         Partially fixed  ⚠️

P1 Issues (6)
├── dataSliceMut              Not fixed        ❌
├── Hardcoded Homebrew        Fixed            ✅
├── EagerContext leak         Not fixed        ❌
├── Attention mask            Pending          ❓
├── NN tests                  Partially fixed  ⚠️
└── golden test               Partially fixed  ⚠️

P2 Issues (4)
├── insertion sort            Not fixed        ❌
├── allocator misleading      Not fixed        ❌
├── ops duplication           Not fixed        ❌
└── scalar ignores dtype      Pending          ❓
```

## 9.3 New Issues (Discovered in This Analysis)

| New Issue | Severity | Location | Description |
|--------|--------|------|------|
| Prompt Cache type safety vulnerability | **P0** | `prompt_cache.zig:74` | `@ptrCast` assumes all caches are StandardKVCache |
| DistributedGroup deinit empty | P1 | `distributed.zig:83` | Resource leak |
| ModelPool vtable null risk | P2 | `model_pool.zig:66` | Model resources may not be released |
| EagleDrafter simplified implementation | P2 | `speculative.zig:146` | Only single token draft effective |
| `strides()` 64-bit assumption | P2 | `array.zig` | Truncation risk on 32-bit platforms |

## 9.4 Deviation Analysis

**Largest deviation**: the project claims NN layer GPU-ization and `@constCast` removal are "fully complete", but `nn.zig` still has 34 `dataSliceMut` (indirectly through `@constCast`), and `minimax.zig` and `deepseek_v4.zig` still have direct `@constCast`.

Possible causes:
1. `activations.zig` GPU-ization was mistaken for "all NN layers" being fixed
2. `dataSliceMut` call statistics were overlooked during the fix
3. New model files (`minimax.zig`, `deepseek_v4.zig`) introduced new `@constCast`
