# Requirements: Stream Mode Expert Loading Correctness

## Overview

DeepSeek V4 stream mode (on-demand expert loading) produces incorrect output despite no crashes or NaN values. The model generates nonsensical tokens (e.g., Korean characters for "What is 2+2?") indicating a fundamental computation error in the expert weight usage pipeline.

## User Stories

### R1: Correct Expert Weight Routing
**As a** developer using stream mode  
**I want** the router-selected experts to be correctly loaded and used in computation  
**So that** the model generates semantically correct output

**Acceptance Criteria:**
- Given a prompt "What is 2+2?"
- When generating with stream mode
- Then the output should be mathematically relevant (e.g., "4", "The answer is 4")
- And NOT produce unrelated content (Korean text, random words)

### R2: Remap Correctness
**As a** stream mode implementation  
**I want** the expert ID remapping to correctly map global IDs to local loaded indices  
**So that** gatherQmm operations use the correct expert weights

**Acceptance Criteria:**
- Given router selects experts [87, 0, 31, 0, 22, 0]
- And unique experts loaded are [0, 22, 31, 87]
- When building remap: remap[0]=0, remap[22]=1, remap[31]=2, remap[87]=3
- Then mlx_take(remap, indices) should produce [3, 0, 2, 0, 1, 0]
- And NOT produce [3, 2, 1, 0, 0, 0] or other incorrect sequences

### R3: Numerical Equivalence with Preload Mode
**As a** user of mlx-zig  
**I want** stream mode to produce identical outputs to preload mode (within numerical tolerance)  
**So that** I can choose between memory efficiency (stream) and speed (preload) without correctness concerns

**Acceptance Criteria:**
- Given the same prompt, seed, and model
- When comparing stream mode vs preload mode outputs
- Then logits should match within cosine similarity > 0.9999
- And generated tokens should be identical

### R4: MLX Array Memory Layout Compatibility
**As a** stream mode implementation  
**I want** to correctly handle MLX array memory layouts when reading indices  
**So that** dataSlice() and mlx_take() operations are consistent

**Acceptance Criteria:**
- Given a 2D indices array [N, topk]
- When reading via dataSlice(u32)
- Then the order should match the order used by mlx_take()
- And manual verification of remap[indices[i]] should equal remapped[i]

## Non-Goals

- Performance optimization (LRU cache, partial tensor loading) — deferred to future work
- Supporting models other than DeepSeek V4 — focus on correctness first
- Preload mode improvements — preload mode is already correct

## Success Metrics

1. **Correctness**: Stream mode generates semantically correct output for 10 test prompts
2. **Consistency**: Stream mode output matches preload mode output (token-level) for 5 test cases
3. **Remap Verification**: All manual remap checks pass (expected == actual) for 100 random expert selections

## Dependencies

- Existing: `src/models/expert_stream.zig` (stream mode implementation)
- Existing: `src/models/expert_preload.zig` (reference correct implementation)
- Existing: `src/models/deepseek_v4.zig` (DSV4SwitchGLU reference)
- Python reference: `vmlx/vmlx_engine/utils/smelt_loader.py` (TurboRouteWrapper)

## Open Questions

1. Is the issue in remap construction, mlx_take usage, or dataSlice reading?
2. Does MLX use row-major or column-major layout for 2D arrays?
3. Should we flatten indices before remapping to avoid layout issues?
4. Is there a mismatch between how Python vmlx and Zig mlx-zig handle array indexing?
