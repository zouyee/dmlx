# Document Layer Decision Tree

Choose the right document type before writing. This prevents over-documentation or under-documentation.

## Quick Decision Tree

```
Is this a technical decision between multiple options?
├── YES → Is the decision already made?
│   ├── NO  → Write RFC (explore options, gather feedback)
│   └── YES → Write ADR (record the decision and why)
└── NO → Is this designing a feature/system?
    ├── YES → Is it complex (affects >3 files or new component)?
    │   ├── YES → Write Kiro Spec (.kiro/specs/<feature>/)
    │   └── NO  → Add section to existing design.md or write Tech Spec
    └── NO → Is this tracking implementation progress?
        ├── YES → Write/update tasks.md
        └── NO → Is this documenting a known issue/fix?
            ├── YES → Write troubleshooting doc
            └── NO → Commit message or inline comment is enough
```

## Document Types Compared

| Aspect | RFC | ADR | Kiro Spec | Tech Spec | Troubleshooting |
|---|---|---|---|---|---|
| **Purpose** | Explore options | Record decision | Design complete feature | Specify implementation details | Map symptoms to fixes |
| **Timing** | Before decision | Right after decision | Before implementation | Just before coding | Continuous |
| **Audience** | Team, stakeholders | Current/future team | Engineers + AI agents | Implementing engineer | Debugger |
| **Length** | 2–10 pages | 1–2 pages | 5–20 pages | 2–5 pages | 1 page per symptom |
| **Lifespan** | Archived after decision | Permanent | Updated during project | Until implementation done | Living document |
| **Location** | `docs/rfcs/` | `docs/adrs/` | `.kiro/specs/<feature>/` | Inside design.md or sub-doc | `docs/troubleshooting/` |

## When to Use Each

### RFC (Request for Comments)

Use when:
- Evaluating new technology adoption (e.g., "Should we use PagedAttention or standard KV cache?")
- Architecture changes with cross-team impact
- Process changes affecting multiple developers

**Not** for: bug fixes, minor version upgrades, implementation details.

**Template location**: Create `docs/rfcs/NNNN-title.md` where NNNN is the next number.

**Required sections**:
1. Summary (1 paragraph)
2. Motivation (why now?)
3. Detailed design (primary proposal)
4. Alternatives considered (at least 2)
5. Trade-offs table
6. Open questions

### ADR (Architecture Decision Record)

Use when:
- A significant technical choice is finalized
- You want future developers to understand "why not X"
- The decision has consequences that constrain future choices

**Not** for: obvious choices (e.g., "use the project's existing formatter").

**Template location**: Create `docs/adrs/NNNN-title.md`.

**Required sections**:
1. Status (Proposed / Accepted / Deprecated / Superseded)
2. Context (what forced this decision)
3. Decision (what we chose)
4. Consequences (positive + negative + risks)
5. Alternatives considered (brief, with why rejected)

### Kiro Spec

Use when:
- Building a new feature or capability
- The work affects multiple components or files (>3 files)
- You need structured tracking across requirements → design → tasks

**Not** for: one-line fixes, trivial refactors, pure exploration.

**Template**: See `kiro-spec-templates.md`.

### Tech Spec

Use when:
- The Kiro Spec design is approved and you need implementation-level detail
- Defining API endpoint signatures, request/response schemas
- Specifying deployment strategy or migration steps

**Location**: Either as a section within `design.md` (for simple cases) or as a `design-<component>.md` sub-document (for complex cases).

### Troubleshooting Doc

Use when:
- A bug is fixed and the symptom-to-root-cause path is non-trivial
- The same issue recurs and the fix is not obvious from code alone
- Multiple checkpoints (spec + code locations) are needed to diagnose

**Location**: `docs/troubleshooting/<feature-or-component>.md`

## Examples from This Project

| Work | Correct Document | Why |
|---|---|---|
| "Add PagedAttention KV cache" | Kiro Spec | New component, affects scheduler + server + kvcache |
| "Should we use continuous batching or static batching?" | RFC | Architectural choice with performance implications |
| "We chose continuous batching" | ADR | Record the decision and its consequences |
| "How does the scheduler allocate blocks?" | Tech Spec (sub-doc) | Implementation detail within the larger spec |
| "Requests stuck in waiting queue" | Troubleshooting | Symptom → spec property → code checkpoints |
| "Fix typo in error message" | None | Commit message sufficient |
