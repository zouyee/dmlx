# Remaining Fixes Plan
# Date: 2026-05-04

## Fix Items

| # | Issue | Priority | Approach | Risk | Status |
|---|-------|----------|----------|------|--------|
| 1 | prompt_cache type safety | P0 | Plan B: vtable pointer comparison assertion | Low | Pending |
| 2 | Gate float32 | P1 | logits.astype(f32) before scoring | Low | Pending |
| 3 | Attention mask return_array | P1 | Build explicit mask with sliding window during prefill | Medium | Pending |
| 4 | dataSliceMut CoW | P1 | Add debug assert ref_count==1 | Low | Pending |

Not fixing:
- Server batch forward (P1): Architecture-level refactor, out of scope for this round
