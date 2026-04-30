---
name: doc-driven-dev
description: Guide AI agents through a document-driven development workflow using Kiro-style specs with lightweight enhancements. Use when the user needs to create, update, or execute specifications for complex features, manage R&D progress through documents, structure implementation around .kiro/specs/ artifacts, or troubleshoot issues by consulting specs before code. Triggers on phrases like "create spec", "write requirements", "design doc", "feature plan", "doc-driven", "Kiro spec", or when the user references .kiro/specs/ directory.
---

# Document-Driven Development Skill

Enforce a structured requirements → design → tasks → implementation workflow with lightweight enhancements on top of Kiro SDD. Specifications are the primary artifact; code is a generated side effect.

## Core Principle

**Read the spec before touching the code.** When asked to implement, modify, or debug a feature, first check `.kiro/specs/` for an existing spec. If none exists, create one before writing implementation code.

## Document Layer Decision Tree

See `references/doc-layer-decision-tree.md` for the full decision matrix. Quick guide:

| Situation | Document Type | Location |
|---|---|---|
| Explore multiple solutions before deciding | RFC | `docs/rfcs/NNNN-title.md` |
| Record why a decision was made | ADR | `docs/adrs/NNNN-title.md` |
| Design a complete feature (this skill's main focus) | Kiro Spec | `.kiro/specs/<feature>/` |
| Define API or deployment details for implementers | Tech Spec | Inside `design.md` section or sub-doc |
| Track R&D progress | Tasks | `.kiro/specs/<feature>/tasks.md` |

**Rule**: If the change affects more than 3 files or introduces a new component, write a Kiro Spec first.

## Kiro Spec Structure

Every feature spec lives in `.kiro/specs/<feature-name>/` and contains three files:

```
.kiro/specs/<feature-name>/
├── requirements.md   # What to build: user stories + acceptance criteria
├── design.md         # How to build: architecture, interfaces, data models
└── tasks.md          # Execution plan: discrete, trackable tasks
```

Use the templates in `references/kiro-spec-templates.md`.

## Enhanced Layer (Beyond Standard Kiro)

### 1. Three-Layer Documentation Model

Align documentation with code at three layers of abstraction:

| Layer | What to document | What to skip | Doc location |
|-------|------------------|--------------|--------------|
| **L1 — Architecture** | Interfaces, signatures, data structures, component boundaries | Implementation details | `design.md` or `design-<component>.md` |
| **L2 — Algorithm** | State machines, invariants, key algorithms, decision rationale | Line-by-line logic | `design-<component>.md` |
| **L3 — Implementation** | Inline comments for tricky code, TODOs, edge cases | — | Source code comments only |

**Rule**: L1 must align with code signatures. L2 should align with actual behavior. L3 never aligns — it lives in code.

### 2. Sub-Documents for Large Modules

When `design.md` exceeds 400 lines or a single component needs deep design, split it into sub-documents referenced from `design.md`:

```markdown
<!-- sub-doc: design-server.md -->
See [Server Design](design-server.md) for HTTP lifecycle, SSE streaming,
and scheduler integration details.
```

**Naming convention**: `design-<component>.md` in the same `.kiro/specs/<feature>/` directory.

**Limit**: Keep `design.md` as the integration layer. Sub-docs contain deep implementation details of one component only. Do not nest sub-docs (no sub-sub-docs).

### 3. Module-Document Mapping Table

Maintain a mapping table in `design.md` (or a dedicated `docs/module-map.md`) so developers can quickly find the design document for any source file:

```markdown
| Source File | Design Document | Layer |
|-------------|-----------------|-------|
| `src/server.zig` | [Server](design-server.md) | L1/L2 |
| `src/batch_builder.zig` | [Batch Builder](design-batch-builder.md) | L1/L2 |
```

**Rule**: When creating a new sub-doc, add its entry to the mapping table before marking the task complete.

### 4. Troubleshooting Directory

Maintain `docs/troubleshooting/<feature>.md` as a living document mapping symptoms to spec sections and code checkpoints:

```markdown
## Scheduler: Requests stuck in waiting queue

**Spec reference**: `design.md` §3.1 — Block Conservation Property
**Code checkpoints**:
1. `src/scheduler.zig:46` — `canAllocate()` logic
2. `src/scheduler.zig:284` — waiting queue promotion
3. `src/kvcache/paged.zig:89` — `freeCount()` accuracy
```

**Rule**: When debugging a component, read its troubleshooting doc first. If the symptom is not listed, add it after fixing the bug.

### 5. Task-Status Sync Rules

See `references/task-sync-rules.md` for the full protocol. Summary:

| Code State | Task Markdown | Meaning |
|---|---|---|
| Not started | `- [ ] 1.1 ...` | Pending |
| In progress | `- [-] 1.1 ...` | Partially done / blocked |
| Implemented, tests passing | `- [x] 1.1 ...` | Complete |
| Implemented, spec needs update | `- [x] 1.1 ...` + `<!-- NOTE: spec gap on X -->` | Complete with known doc debt |

**Post-implementation rule**: After completing a task, scan the spec for gaps between design and actual implementation. If the code diverged from the design, add a `<!-- NOTE: ... -->` in `tasks.md` and update `design.md` within the same PR.

## Workflow Phases

### Phase 1: Requirements

1. Check `.kiro/specs/` for existing specs covering this work
2. If none exists, create `.kiro/specs/<feature>/requirements.md`
3. Write user stories with acceptance criteria in Given/When/Then format
4. Number every requirement (R1, R2, ...)
5. Define explicit non-goals

**Exit criteria**: A developer not in the conversation can implement the feature without clarifying questions.

### Phase 2: Design

1. Create `.kiro/specs/<feature>/design.md`
2. Include system architecture diagram (mermaid)
3. Define component interfaces with type signatures
4. Document data models and persistence formats
5. List correctness properties that must hold
6. Document error handling strategy
7. If a component needs >100 lines of design detail, extract to `design-<component>.md`
8. Add new sub-docs to the module-document mapping table

**Exit criteria**: The design can be reviewed and approved without reading implementation code.

### Phase 3: Tasks

1. Create `.kiro/specs/<feature>/tasks.md`
2. Break design into discrete tasks, each completable in one focused session
3. Each task references its requirement numbers
4. Each task has explicit acceptance criteria
5. Order tasks by dependency, not perceived importance

**Exit criteria**: A task list where executing items in order produces the design.

### Phase 4: Implement

1. Pick the next uncompleted task from `tasks.md`
2. Load the relevant spec sections and source files into context
3. Implement the task
4. Update the task status in `tasks.md`
5. If implementation diverged from design, add a NOTE and update `design.md`
6. Repeat until all tasks complete

**Rule**: Do not implement anything not in the spec. If a missing requirement is discovered, stop and update the spec first.

## Troubleshooting Protocol

When asked to debug or investigate an issue:

1. **Check `.kiro/specs/` first** — Is there a spec for the affected component?
2. **Read `design.md` Correctness Properties** — Does the failure violate a documented property?
3. **Read `docs/troubleshooting/<feature>.md`** — Is this a known symptom?
4. **Read `tasks.md` for incomplete items** — Is this a known gap?
5. **Only then read source code** — Use the spec's component interface definitions to narrow the search

See `references/troubleshooting-guide.md` for detailed examples.

## Anti-Patterns

| Anti-Pattern | Correction |
|---|---|
| Coding before spec exists | Stop. Write requirements.md → design.md → tasks.md first. |
| Spec as post-hoc documentation | A spec written after code is not a spec. Relabel it as `docs/archived/`. |
| Vague acceptance criteria | "The system should work well" is not acceptable. Use measurable criteria. |
| Gold-plating beyond spec | If it is not in the spec, file a new spec. Do not add it to the current task. |
| Tasks.md never updated | Tasks status must reflect code reality. Outdated tasks are worse than no tasks. |

## When to Skip This Workflow

| Situation | Action |
|---|---|
| Typo fix, rename variable, formatting | Skip spec. Edit code directly. |
| Simple refactor touching ≤2 files | Skip spec. Document in commit message. |
| Prototype / spike / exploration | Use "Vibe" mode. If the spike succeeds, write a spec before productionizing. |
| Emergency hotfix | Fix first, write retroactive spec within 24 hours. |
