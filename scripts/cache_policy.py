#!/usr/bin/env python3
"""Deterministic offline cache-policy accounting for validated routing traces.

This module never mutates runtime state. It consumes a total-ordered stream of
numeric route events and compares bounded per-layer user-cache policies using
application-visible expert pread bytes as the physical-byte objective.
"""

from __future__ import annotations

from collections import Counter, OrderedDict, defaultdict, deque
from dataclasses import dataclass
from typing import Any, Iterable

PHASE_NAMES = {0: "prefill", 1: "decode", 2: "verify"}
CACHE_HOT = 1
CACHE_PREFETCH = 2
CACHE_BATCH_POOL = 3
CACHE_COLD = 4
VALID_CACHE_STATES = {CACHE_HOT, CACHE_PREFETCH, CACHE_BATCH_POOL, CACHE_COLD}


class CachePolicyError(ValueError):
    """Raised when an event stream or policy geometry is ambiguous."""


@dataclass(frozen=True)
class RouteEvent:
    order_index: int
    request_id: int
    event_seq: int
    phase: int
    token_position: int
    layer: int
    expert_ids: tuple[int, ...]
    cache_states: tuple[int, ...]
    route_bytes: tuple[int, ...]


def _new_metrics() -> dict[str, int]:
    return {
        "route_records": 0,
        "expert_routes": 0,
        "hits": 0,
        "misses": 0,
        "read_bytes": 0,
        "admissions": 0,
        "evictions": 0,
        "slot_churn": 0,
    }


def _record(
    result: dict[str, Any],
    event: RouteEvent,
    *,
    hits: int,
    misses: int,
    read_bytes: int,
    admissions: int = 0,
    evictions: int = 0,
) -> None:
    phase = PHASE_NAMES[event.phase]
    layer = str(event.layer)
    buckets = (
        result["totals"],
        result["by_phase"].setdefault(phase, _new_metrics()),
        result["by_layer"].setdefault(layer, _new_metrics()),
        result["by_phase_layer"].setdefault(phase, {}).setdefault(
            layer, _new_metrics()
        ),
    )
    for bucket in buckets:
        bucket["route_records"] += 1
        bucket["expert_routes"] += len(event.expert_ids)
        bucket["hits"] += hits
        bucket["misses"] += misses
        bucket["read_bytes"] += read_bytes
        bucket["admissions"] += admissions
        bucket["evictions"] += evictions
        bucket["slot_churn"] += evictions


def _finalize_metrics(metrics: dict[str, int]) -> dict[str, Any]:
    result: dict[str, Any] = dict(metrics)
    accesses = metrics["expert_routes"]
    result["hit_rate"] = metrics["hits"] / accesses if accesses else None
    result["bytes_per_expert_route"] = (
        metrics["read_bytes"] / accesses if accesses else None
    )
    return result


def _finalize_result(result: dict[str, Any]) -> dict[str, Any]:
    result["totals"] = _finalize_metrics(result["totals"])
    result["by_phase"] = {
        phase: _finalize_metrics(metrics)
        for phase, metrics in sorted(result["by_phase"].items())
    }
    result["by_layer"] = {
        layer: _finalize_metrics(metrics)
        for layer, metrics in sorted(
            result["by_layer"].items(), key=lambda item: int(item[0])
        )
    }
    result["by_phase_layer"] = {
        phase: {
            layer: _finalize_metrics(metrics)
            for layer, metrics in sorted(
                layers.items(), key=lambda item: int(item[0])
            )
        }
        for phase, layers in sorted(result["by_phase_layer"].items())
    }
    return result


def _new_result(
    name: str,
    role: str,
    *,
    eligible_for_online_ab: bool,
    assumptions: list[str],
    resident_slots: int,
    resident_bytes: int,
) -> dict[str, Any]:
    return {
        "name": name,
        "role": role,
        "eligible_for_online_ab": eligible_for_online_ab,
        "assumptions": assumptions,
        "memory": {
            "resident_slots": resident_slots,
            "resident_bytes": resident_bytes,
            "startup_preload_bytes_in_request_objective": 0,
        },
        "totals": _new_metrics(),
        "by_phase": {},
        "by_layer": {},
        "by_phase_layer": {},
    }


def _validate_events(
    events: list[RouteEvent],
    *,
    n_layers: int,
    n_experts: int,
    expert_size_bytes: int,
) -> None:
    if not events:
        raise CachePolicyError("policy replay requires at least one route event")
    if type(n_layers) is not int or n_layers <= 0:
        raise CachePolicyError("n_layers must be positive")
    if type(n_experts) is not int or n_experts <= 0:
        raise CachePolicyError("n_experts must be positive")
    if type(expert_size_bytes) is not int or expert_size_bytes <= 0:
        raise CachePolicyError("expert_size_bytes must be positive")
    for expected_order, event in enumerate(events):
        if event.order_index != expected_order:
            raise CachePolicyError("route order_index must be contiguous and total-ordered")
        if event.phase not in PHASE_NAMES:
            raise CachePolicyError(f"event {expected_order} has an invalid phase")
        if event.layer < 0 or event.layer >= n_layers:
            raise CachePolicyError(f"event {expected_order} has an invalid layer")
        width = len(event.expert_ids)
        if width == 0 or len(event.cache_states) != width or len(event.route_bytes) != width:
            raise CachePolicyError(f"event {expected_order} has inconsistent route arrays")
        if len(set(event.expert_ids)) != width:
            raise CachePolicyError(f"event {expected_order} repeats an expert within a route")
        for expert, state, byte_count in zip(
            event.expert_ids, event.cache_states, event.route_bytes
        ):
            if expert < 0 or expert >= n_experts:
                raise CachePolicyError(f"event {expected_order} has an invalid expert")
            if state not in VALID_CACHE_STATES:
                raise CachePolicyError(f"event {expected_order} has an invalid cache state")
            if byte_count not in (0, expert_size_bytes):
                raise CachePolicyError(f"event {expected_order} has invalid route bytes")
            if state == CACHE_COLD and byte_count != expert_size_bytes:
                raise CachePolicyError(f"event {expected_order} cold route lacks a full read")
            if state != CACHE_COLD and byte_count != 0:
                raise CachePolicyError(f"event {expected_order} non-cold route has read bytes")


def _capacities(
    n_layers: int,
    n_experts: int,
    capacity: int,
    full_resident_layers: int,
) -> list[int]:
    if type(capacity) is not int or capacity <= 0 or capacity > n_experts:
        raise CachePolicyError("capacity must be in [1, n_experts]")
    if (
        type(full_resident_layers) is not int
        or full_resident_layers < 0
        or full_resident_layers > n_layers
    ):
        raise CachePolicyError("full_resident_layers is outside the model")
    return [
        n_experts if layer < full_resident_layers else capacity
        for layer in range(n_layers)
    ]


def _observed_hot_seed(
    events: list[RouteEvent],
    capacities: list[int],
    n_experts: int,
    full_resident_layers: int,
) -> list[set[int]]:
    hot: list[set[int]] = [set() for _ in capacities]
    non_hot: list[set[int]] = [set() for _ in capacities]
    for event in events:
        for expert, state in zip(event.expert_ids, event.cache_states):
            if state == CACHE_HOT:
                hot[event.layer].add(expert)
            else:
                non_hot[event.layer].add(expert)
    for layer, capacity in enumerate(capacities):
        if layer < full_resident_layers:
            if non_hot[layer]:
                raise CachePolicyError(
                    f"full-resident layer {layer} contains an observed non-hot route"
                )
            hot[layer] = set(range(n_experts))
            continue
        inconsistent = hot[layer] & non_hot[layer]
        if inconsistent:
            raise CachePolicyError(
                f"layer {layer} has experts observed as both HOT and non-HOT: "
                f"{sorted(inconsistent)}"
            )
        if len(hot[layer]) > capacity:
            raise CachePolicyError(
                f"layer {layer} exposes {len(hot[layer])} hot experts for capacity {capacity}"
            )
    return hot


def _seed_with_anonymous_slots(
    hot: list[set[int]], capacities: list[int]
) -> list[list[int]]:
    result: list[list[int]] = []
    for layer, (members, capacity) in enumerate(zip(hot, capacities)):
        real = sorted(members)
        anonymous = [-(1 + layer * 1_000_000 + slot) for slot in range(capacity - len(real))]
        result.append(anonymous + real)
    return result


def _resident_geometry(
    capacities: list[int], expert_size_bytes: int
) -> tuple[int, int]:
    slots = sum(capacities)
    return slots, slots * expert_size_bytes


def _observed_fixed_hot(
    events: list[RouteEvent], resident_slots: int, resident_bytes: int
) -> dict[str, Any]:
    result = _new_result(
        "fixed_hot_observed",
        "baseline",
        eligible_for_online_ab=False,
        assumptions=[
            "HOT is a user-cache hit; every other observed state is a miss",
            "read_bytes use the trace's actual synchronous route bytes",
            "resident membership never changes and startup preload is excluded",
        ],
        resident_slots=resident_slots,
        resident_bytes=resident_bytes,
    )
    for event in events:
        hits = sum(state == CACHE_HOT for state in event.cache_states)
        _record(
            result,
            event,
            hits=hits,
            misses=len(event.expert_ids) - hits,
            read_bytes=sum(event.route_bytes),
        )
    return _finalize_result(result)


def _trust_os(
    events: list[RouteEvent], expert_size_bytes: int
) -> dict[str, Any]:
    result = _new_result(
        "trust_os_no_user_cache",
        "reference",
        eligible_for_online_ab=True,
        assumptions=[
            "no user-space resident expert slots",
            "every routed expert issues one full application-visible pread",
            "OS page-cache hits are intentionally not claimed as zero-byte reads",
        ],
        resident_slots=0,
        resident_bytes=0,
    )
    for event in events:
        misses = len(event.expert_ids)
        _record(
            result,
            event,
            hits=0,
            misses=misses,
            read_bytes=misses * expert_size_bytes,
        )
    return _finalize_result(result)


def _lru(
    events: list[RouteEvent],
    seeds: list[list[int]],
    capacities: list[int],
    resident_slots: int,
    resident_bytes: int,
    expert_size_bytes: int,
) -> dict[str, Any]:
    result = _new_result(
        "lru_observed_hot_seeded",
        "candidate",
        eligible_for_online_ab=True,
        assumptions=[
            "starts from the observed fixed-hot membership and anonymous occupied slots",
            "route hits are classified atomically before rank-ordered admissions",
            "least-recent resident is replaced on each miss",
        ],
        resident_slots=resident_slots,
        resident_bytes=resident_bytes,
    )
    caches = [OrderedDict((expert, None) for expert in seed) for seed in seeds]
    for event in events:
        cache = caches[event.layer]
        snapshot = set(cache)
        hit_ids = [expert for expert in event.expert_ids if expert in snapshot]
        miss_ids = [expert for expert in event.expert_ids if expert not in snapshot]
        for expert in hit_ids:
            cache.move_to_end(expert)
        evictions = 0
        for expert in miss_ids:
            if len(cache) >= capacities[event.layer]:
                cache.popitem(last=False)
                evictions += 1
            cache[expert] = None
        _record(
            result,
            event,
            hits=len(hit_ids),
            misses=len(miss_ids),
            read_bytes=len(miss_ids) * expert_size_bytes,
            admissions=len(miss_ids),
            evictions=evictions,
        )
    return _finalize_result(result)


def _lfu(
    events: list[RouteEvent],
    seeds: list[list[int]],
    capacities: list[int],
    resident_slots: int,
    resident_bytes: int,
    expert_size_bytes: int,
) -> dict[str, Any]:
    result = _new_result(
        "lfu_observed_hot_seeded",
        "candidate",
        eligible_for_online_ab=True,
        assumptions=[
            "starts from the observed fixed-hot membership and anonymous occupied slots",
            "route hits are classified atomically before rank-ordered admissions",
            "eviction key is frequency, then last access, then expert id",
        ],
        resident_slots=resident_slots,
        resident_bytes=resident_bytes,
    )
    caches: list[dict[int, list[int]]] = []
    clock = 0
    for seed in seeds:
        state: dict[int, list[int]] = {}
        for index, expert in enumerate(seed):
            state[expert] = [0, -len(seed) + index]
        caches.append(state)
    for event in events:
        cache = caches[event.layer]
        snapshot = set(cache)
        hit_ids = [expert for expert in event.expert_ids if expert in snapshot]
        miss_ids = [expert for expert in event.expert_ids if expert not in snapshot]
        for expert in hit_ids:
            clock += 1
            cache[expert][0] += 1
            cache[expert][1] = clock
        evictions = 0
        for expert in miss_ids:
            if len(cache) >= capacities[event.layer]:
                victim = min(
                    cache,
                    key=lambda item: (cache[item][0], cache[item][1], item),
                )
                del cache[victim]
                evictions += 1
            clock += 1
            cache[expert] = [1, clock]
        _record(
            result,
            event,
            hits=len(hit_ids),
            misses=len(miss_ids),
            read_bytes=len(miss_ids) * expert_size_bytes,
            admissions=len(miss_ids),
            evictions=evictions,
        )
    return _finalize_result(result)


def _static_frequency_oracle(
    events: list[RouteEvent],
    capacities: list[int],
    n_experts: int,
    resident_slots: int,
    resident_bytes: int,
    expert_size_bytes: int,
) -> dict[str, Any]:
    result = _new_result(
        "static_frequency_oracle",
        "isolated_oracle_candidate",
        eligible_for_online_ab=False,
        assumptions=[
            "selects each layer's highest-frequency experts using the evaluated trace",
            "future-aware membership makes this a ceiling, not deployable evidence",
            "misses bypass the immutable static set; startup preload is excluded",
        ],
        resident_slots=resident_slots,
        resident_bytes=resident_bytes,
    )
    frequencies: list[Counter[int]] = [Counter() for _ in capacities]
    for event in events:
        frequencies[event.layer].update(event.expert_ids)
    residents: list[set[int]] = []
    for layer, capacity in enumerate(capacities):
        if capacity == n_experts:
            residents.append(set(range(n_experts)))
            continue
        ranked = sorted(
            frequencies[layer], key=lambda expert: (-frequencies[layer][expert], expert)
        )
        residents.append(set(ranked[:capacity]))
    for event in events:
        hits = sum(expert in residents[event.layer] for expert in event.expert_ids)
        misses = len(event.expert_ids) - hits
        _record(
            result,
            event,
            hits=hits,
            misses=misses,
            read_bytes=misses * expert_size_bytes,
        )
    return _finalize_result(result)


def _belady_lower_bound(
    events: list[RouteEvent],
    seeds: list[list[int]],
    capacities: list[int],
    resident_slots: int,
    resident_bytes: int,
    expert_size_bytes: int,
) -> dict[str, Any]:
    result = _new_result(
        "belady_same_seed_lower_bound",
        "theoretical_lower_bound",
        eligible_for_online_ab=False,
        assumptions=[
            "starts from the same observed-hot/anonymous residency as dynamic candidates",
            "future-aware atomic retention may bypass a miss and is not implementable online",
            "retains residents with nearest next route use after each atomic event",
        ],
        resident_slots=resident_slots,
        resident_bytes=resident_bytes,
    )
    future: dict[tuple[int, int], deque[int]] = defaultdict(deque)
    for event in events:
        for expert in event.expert_ids:
            future[(event.layer, expert)].append(event.order_index)
    caches = [set(seed) for seed in seeds]
    for event in events:
        cache = caches[event.layer]
        snapshot = set(cache)
        hits = sum(expert in snapshot for expert in event.expert_ids)
        misses = len(event.expert_ids) - hits
        for expert in event.expert_ids:
            queue = future[(event.layer, expert)]
            if not queue or queue[0] != event.order_index:
                raise CachePolicyError("future-use index is inconsistent with event order")
            queue.popleft()
        candidates = cache | set(event.expert_ids)

        def retention_key(expert: int) -> tuple[float, int, int]:
            queue = future.get((event.layer, expert))
            next_use = float(queue[0]) if queue else float("inf")
            return (next_use, 1 if expert < 0 else 0, expert)

        retained = set(sorted(candidates, key=retention_key)[: capacities[event.layer]])
        removed = len(cache - retained)
        added = len(retained - cache)
        caches[event.layer] = retained
        _record(
            result,
            event,
            hits=hits,
            misses=misses,
            read_bytes=misses * expert_size_bytes,
            admissions=added,
            evictions=removed,
        )
    return _finalize_result(result)


def compare_policies(
    events: Iterable[RouteEvent],
    *,
    n_layers: int,
    n_experts: int,
    capacity: int,
    expert_size_bytes: int,
    full_resident_layers: int = 0,
) -> dict[str, Any]:
    """Compare all Task 6.2 policies on one validated route stream."""
    event_list = list(events)
    _validate_events(
        event_list,
        n_layers=n_layers,
        n_experts=n_experts,
        expert_size_bytes=expert_size_bytes,
    )
    capacities = _capacities(
        n_layers, n_experts, capacity, full_resident_layers
    )
    observed_hot = _observed_hot_seed(
        event_list, capacities, n_experts, full_resident_layers
    )
    seeds = _seed_with_anonymous_slots(observed_hot, capacities)
    resident_slots, resident_bytes = _resident_geometry(
        capacities, expert_size_bytes
    )

    policies = [
        _observed_fixed_hot(event_list, resident_slots, resident_bytes),
        _lru(
            event_list,
            seeds,
            capacities,
            resident_slots,
            resident_bytes,
            expert_size_bytes,
        ),
        _lfu(
            event_list,
            seeds,
            capacities,
            resident_slots,
            resident_bytes,
            expert_size_bytes,
        ),
        _trust_os(event_list, expert_size_bytes),
        _static_frequency_oracle(
            event_list,
            capacities,
            n_experts,
            resident_slots,
            resident_bytes,
            expert_size_bytes,
        ),
        _belady_lower_bound(
            event_list,
            seeds,
            capacities,
            resident_slots,
            resident_bytes,
            expert_size_bytes,
        ),
    ]
    baseline = policies[0]["totals"]
    baseline_bytes = baseline["read_bytes"]
    baseline_hit_rate = baseline["hit_rate"]
    for policy in policies:
        totals = policy["totals"]
        policy["delta_vs_fixed_hot"] = {
            "read_bytes": totals["read_bytes"] - baseline_bytes,
            "read_bytes_percent": (
                (totals["read_bytes"] - baseline_bytes) * 100.0 / baseline_bytes
                if baseline_bytes
                else None
            ),
            "hit_rate_percentage_points": (
                (totals["hit_rate"] - baseline_hit_rate) * 100.0
                if totals["hit_rate"] is not None and baseline_hit_rate is not None
                else None
            ),
            "slot_churn": totals["slot_churn"] - baseline["slot_churn"],
        }

    candidates = [
        policy
        for policy in policies
        if policy["role"] in {"candidate", "reference", "isolated_oracle_candidate"}
    ]
    candidates.sort(
        key=lambda policy: (
            policy["totals"]["read_bytes"],
            policy["totals"]["slot_churn"],
            policy["name"],
        )
    )
    ranking = [
        {
            "rank": rank,
            "name": policy["name"],
            "role": policy["role"],
            "eligible_for_online_ab": policy["eligible_for_online_ab"],
            "read_bytes": policy["totals"]["read_bytes"],
            "hit_rate": policy["totals"]["hit_rate"],
            "slot_churn": policy["totals"]["slot_churn"],
            "read_bytes_delta_vs_fixed_hot": policy["delta_vs_fixed_hot"]["read_bytes"],
        }
        for rank, policy in enumerate(candidates, 1)
    ]
    return {
        "schema_version": 1,
        "kind": "dmlx-exact-q4-cache-policy-comparison",
        "geometry": {
            "n_layers": n_layers,
            "n_experts": n_experts,
            "score_layer_capacity": capacity,
            "full_resident_layers": full_resident_layers,
            "per_layer_capacities": capacities,
            "expert_size_bytes": expert_size_bytes,
            "resident_slots": resident_slots,
            "resident_bytes": resident_bytes,
        },
        "event_semantics": {
            "order": "input order validated by contiguous order_index",
            "route_classification": "atomic before deterministic admissions",
            "physical_byte_objective": "application-visible full expert pread bytes",
            "slot_churn": "resident evictions/replacements",
            "startup_preload": "excluded from request read_bytes; reported as memory",
        },
        "policies": policies,
        "candidate_ranking": ranking,
        "theoretical_lower_bound_policy": "belady_same_seed_lower_bound",
    }


def run_self_test() -> None:
    def event(
        order: int,
        phase: int,
        experts: tuple[int, int],
        states: tuple[int, int],
    ) -> RouteEvent:
        return RouteEvent(
            order_index=order,
            request_id=1,
            event_seq=order,
            phase=phase,
            token_position=order,
            layer=0,
            expert_ids=experts,
            cache_states=states,
            route_bytes=tuple(0 if state == CACHE_HOT else 100 for state in states),
        )

    events = [
        event(0, 0, (0, 1), (CACHE_HOT, CACHE_HOT)),
        event(1, 0, (2, 0), (CACHE_COLD, CACHE_HOT)),
        event(2, 1, (1, 2), (CACHE_HOT, CACHE_COLD)),
        event(3, 1, (3, 2), (CACHE_COLD, CACHE_COLD)),
    ]
    comparison = compare_policies(
        events,
        n_layers=1,
        n_experts=4,
        capacity=2,
        expert_size_bytes=100,
    )
    by_name = {policy["name"]: policy for policy in comparison["policies"]}
    expected = {
        "fixed_hot_observed": (4, 4, 400, 0),
        "lru_observed_hot_seeded": (5, 3, 300, 3),
        "lfu_observed_hot_seeded": (5, 3, 300, 3),
        "trust_os_no_user_cache": (0, 8, 800, 0),
        "static_frequency_oracle": (5, 3, 300, 0),
        "belady_same_seed_lower_bound": (6, 2, 200, 1),
    }
    for name, (hits, misses, read_bytes, churn) in expected.items():
        totals = by_name[name]["totals"]
        assert totals["hits"] == hits, (name, totals)
        assert totals["misses"] == misses, (name, totals)
        assert totals["read_bytes"] == read_bytes, (name, totals)
        assert totals["slot_churn"] == churn, (name, totals)
    assert comparison == compare_policies(
        events,
        n_layers=1,
        n_experts=4,
        capacity=2,
        expert_size_bytes=100,
    )
    assert by_name["belady_same_seed_lower_bound"]["by_phase_layer"]["decode"]["0"]["read_bytes"] == 100
    assert comparison["candidate_ranking"][0]["name"] == "static_frequency_oracle"

    inconsistent = list(events)
    inconsistent[-1] = RouteEvent(
        **{
            **inconsistent[-1].__dict__,
            "expert_ids": (0, 2),
            "cache_states": (CACHE_COLD, CACHE_COLD),
            "route_bytes": (100, 100),
        }
    )
    try:
        compare_policies(
            inconsistent,
            n_layers=1,
            n_experts=4,
            capacity=2,
            expert_size_bytes=100,
        )
    except CachePolicyError:
        pass
    else:
        raise AssertionError("inconsistent observed hot membership was accepted")

    full_resident = compare_policies(
        [event(0, 0, (2, 3), (CACHE_HOT, CACHE_HOT))],
        n_layers=1,
        n_experts=4,
        capacity=2,
        expert_size_bytes=100,
        full_resident_layers=1,
    )
    assert full_resident["geometry"]["resident_slots"] == 4
    assert all(policy["totals"]["misses"] == 0 for policy in full_resident["policies"] if policy["name"] != "trust_os_no_user_cache")

    print("cache policy self-test: PASS")


if __name__ == "__main__":
    run_self_test()
