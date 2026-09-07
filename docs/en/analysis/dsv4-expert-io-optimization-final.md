# DSV4 Expert I/O Optimization — Final Decision

## Decision

`NOT_PROVEN` at baseline Task 1.1. The existing fixed-hot implementation remains the production default. No direct-to-victim candidate code was authorized or implemented, and no promotion run was performed.

## Evidence replay

Three native N=20 fixed-hot benchmark artifacts were retained without selective replacement. All three passed Paris correctness and E2E 7/7, reported zero short reads and complete exact-Q4 producers, stayed below the 44 GiB RSS limit, and recorded consistent expert-I/O totals. `dsv4-expert-io-baseline.json` retains the protocol and benchmark/raw artifact hashes; the historical physical source hashes are missing and therefore explicitly `NOT_PROVEN`.

The frozen R2 contract is conjunctive. The retained artifacts do not contain:

- hashes binding the historical binary, model, tokenizer, packed experts, and collector tools;
- PID/executable identities binding each sample to three independent processes;
- routing-stats seed backup hash and restore events for each process;
- structured pre-start cleanup, verify-none, start/stop, and final no-residual-server lifecycle evidence.

Current filesystem hashes cannot retroactively identify files used by historical processes. Report timestamps and equal routing counters cannot prove clean starts, process independence, or seed restoration. Under R2.3, missing evidence invalidates the complete baseline and cannot be replaced by favorable samples.

## Dependency closure

- Task 1.1: `NOT_PROVEN`.
- Task 2.1 candidate selection: `BLOCKED_BY_REJECTED_DEPENDENCY`.
- Task 3.1 default-off implementation: `BLOCKED_BY_REJECTED_DEPENDENCY`.
- Task 4.1 promotion: `BLOCKED_BY_REJECTED_DEPENDENCY`.

Thresholds and evidence requirements were not relaxed. The failed packed-directory preflight generated no valid sample and is not part of the baseline set. No rerun or sample replacement is authorized by the frozen protocol.

## Operational state

Fixed-hot remains unchanged and default. `NATIVE_EXPERIMENTAL_CACHE_DIRECT_IO` was not added. Full expert preload remains forbidden, and all future server work must continue to use native mode, N=20, packed experts, and one serial `dmlx serve` process.
