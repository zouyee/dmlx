# Task-Status Sync Rules

Keep `tasks.md` aligned with code reality. Outdated tasks are worse than no tasks.

## Status Markers

Use these exact markers in `tasks.md`:

| Marker | Meaning | When to Use |
|---|---|---|
| `- [ ]` | Pending | Task not started |
| `- [-]` | In Progress / Partial | Some sub-tasks done, or blocked waiting on dependency |
| `- [x]` | Complete | Code implemented AND tested AND verified against acceptance criteria |
| `- [x]*` | Complete (optional) | Task was optional and is now done |
| `- [~]` | Obsolete | Task no longer needed (design changed). Add note explaining why. |

## The Two-Way Sync Rule

### Code → Tasks (after implementing)

After completing a task:

1. Mark the task `[x]` in `tasks.md`
2. Scan the task's acceptance criteria. Did the implementation match?
3. If implementation diverged from design, add a `<!-- NOTE -->`:
   ```markdown
   - [x] 5.1 Implement request scheduler
     <!-- NOTE: Changed from priority queue to round-robin because 
          priority inversion caused starvation. See src/scheduler.zig:120 -->
   ```
4. If the divergence is significant (>20% of component behavior), update `design.md` in the same PR

### Tasks → Code (before implementing)

Before starting a task:

1. Read the task and its referenced requirements
2. If requirements are ambiguous, clarify before coding
3. If design.md does not cover the task's scope, pause and extend the design

## Document Backfill Rule

When code reveals a design gap (the implementation needed something the spec didn't anticipate):

| Gap Severity | Action | Timeline |
|---|---|---|
| Minor (single function signature change) | Add NOTE in tasks.md | Same PR |
| Moderate (algorithm different from design) | Update design.md section + add NOTE | Same PR |
| Major (new component not in design) | Write new sub-doc `design-<component>.md` + update design.md architecture | Next PR, before feature ships |

## Audit Trail Convention

Use HTML comments for audit trail entries that should not render in markdown viewers but remain in source:

```markdown
- [x] 3.2 Implement SSE streaming
  <!-- AUDIT: 2026-04-28 — Added keep-alive comments after 
       discovering client timeout at 5s. Requirement R12.3 added. -->
```

## Checkpoint: Before Marking Complete

Run this checklist before changing `[ ]` or `[-]` to `[x]`:

- [ ] Code compiles without warnings
- [ ] Unit tests pass (if test file exists)
- [ ] Acceptance criteria from requirements.md are satisfied
- [ ] If task references a Correctness Property, verify it holds
- [ ] tasks.md status updated
- [ ] If design diverged, NOTE added or design.md updated
- [ ] No `TODO` comments left in new code (convert to tasks.md items if needed)

## Anti-Patterns

| Anti-Pattern | Why It's Wrong | Fix |
|---|---|---|
| Bulk-mark all tasks `[x]` at end | Lose granular progress visibility | Mark tasks as they complete |
| Mark `[x]` when code compiles but tests fail | False progress signal | Wait for tests + criteria |
| Never update tasks.md after initial creation | Document becomes useless | Sync at end of each session |
| Delete obsolete tasks instead of marking `[~]` | Lose history of scope changes | Mark `[~]` with explanation |
