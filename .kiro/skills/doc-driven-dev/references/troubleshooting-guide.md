# Troubleshooting Guide

Protocol for diagnosing issues by reading documents before diving into code.

## The Document-First Diagnostic Flow

When a bug or unexpected behavior is reported, follow this order:

```
1. Identify affected component
   └── Read .kiro/specs/*/design.md — which component owns this behavior?

2. Check correctness properties
   └── Does the failure violate a documented Correctness Property?
   └── If yes, the design assumption is broken → read implementation

3. Check known issues
   └── Read docs/troubleshooting/<component>.md — is this a known symptom?

4. Check task gaps
   └── Read .kiro/specs/*/tasks.md — is there an uncompleted or [-] task
       in this component?

5. Read source code (narrowed by spec)
   └── Use design.md interface definitions to identify files and functions
   └── Use Correctness Property to identify the invariant that failed
```

**Rule**: Spend at least 5 minutes in steps 1–4 before opening source files.

## Reading Specs for Diagnosis

### From design.md

Look for these sections in order:

1. **Correctness Properties** — Does the bug break a documented invariant? If so, the root cause is likely in the code path that should enforce that property.
2. **Error Handling** — Is the error category documented? What handling strategy was specified?
3. **Component Interfaces** — What are the expected inputs/outputs of the failing function? Are the actual values within spec?
4. **Data Models** — Is the state corrupted? Compare actual data structure against the documented model.

### From tasks.md

Look for:
- `[-]` tasks in the affected component — partial implementation is a common source of bugs
- `<!-- NOTE -->` comments — these document known divergences between design and code
- Obsolete `[~]` tasks — scope changes sometimes leave dead code paths

### From requirements.md

Look for:
- Acceptance criteria that should prevent this behavior — if the criteria exist but the bug still happens, the test coverage is missing
- Non-goals — confirm the bug is actually in scope (not a "working as designed" case)

## Writing Troubleshooting Entries

When you fix a non-trivial bug, add an entry to `docs/troubleshooting/<component>.md`:

```markdown
## Symptom: <One-line description>

**Component**: <Name>
**First seen**: <Date or commit>
**Root cause**: <One sentence>

### Spec References

- `design.md` §<section> — <relevant design decision>
- Property <N>: <name> — <why it was violated or not violated>

### Code Checkpoints

1. `<file>:<line>` — <what to check here>
2. `<file>:<line>` — <what to check here>

### Fix Summary

<What was changed and why>

### Prevention

<What spec or test should catch this next time>
```

## Example Entries

### Example 1: Scheduler — Requests stuck in waiting queue

```markdown
## Symptom: Requests remain in waiting queue indefinitely

**Component**: Scheduler / BlockManager
**First seen**: 2026-04-20
**Root cause**: BlockManager.freeCount() under-reported free blocks due to
  not tracking CoW shared blocks correctly.

### Spec References

- `design.md` §3.2 — Block Manager interface
- Property 8: Block Conservation — "sum of free blocks and used blocks SHALL
  always equal the total block count"

### Code Checkpoints

1. `src/kvcache/paged.zig:89` — `freeCount()` calculation
2. `src/scheduler.zig:46` — `canAllocate()` uses freeCount()
3. `src/scheduler.zig:284` — waiting queue promotion logic

### Fix Summary

Changed BlockManager to track ref_count separately from used/allocated state.
Shared blocks now decrement freeCount only once, not per-request.

### Prevention

- Property 8 test should include multi-request CoW scenario
- Add invariant assertion in BlockManager: free + used == total
```

### Example 2: Generation — First token is always EOS

```markdown
## Symptom: Model generates EOS as first token for all prompts

**Component**: Generation API / Sampler
**First seen**: 2026-04-22
**Root cause**: Temperature=0 caused division by zero in softmax, producing
  NaN logits that resolved to token 0 (EOS).

### Spec References

- `design.md` §2.1 — GenerateConfig defines temperature default as 0.8
- Property 3: Generation API Consistency

### Code Checkpoints

1. `src/generation.zig:72` — GenerateConfig default temperature
2. `src/sampling.zig:45` — softmax implementation
3. `src/generation.zig:126` — sampler.sample() call

### Fix Summary

Added temperature clamp: if temperature < 1e-6, use argmax instead of softmax.
Also changed default from 0.0 (test value) back to 0.8 per spec.

### Prevention

- Add sampler test for temperature=0 edge case
- Validate GenerateConfig at construction time
```

## Quick Checklist by Symptom Category

| Symptom Category | First Read | Second Read | Then Check |
|---|---|---|---|
| Crash / panic | `design.md` Error Handling | `tasks.md` for `[-]` items | Stack trace → source |
| Wrong output | `design.md` Correctness Properties | `requirements.md` acceptance criteria | Golden test diff |
| Performance | `design.md` Key Design Decisions | `tasks.md` optimization tasks | Profiler output |
| Memory leak | `design.md` Data Models + Arena pattern | `tasks.md` cleanup tasks | Arena tracking |
| Hang / infinite loop | `design.md` state machines | `tasks.md` loop/termination tasks | Loop conditions |
| Build failure | `requirements.md` build requirements | `design.md` component dependencies | build.zig |
