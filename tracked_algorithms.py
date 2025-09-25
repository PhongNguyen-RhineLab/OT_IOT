import numpy as np
from image_division import image_division


class OperationTracker:
    """Track operations and function calls for algorithm analysis - CORRECTED VERSION"""

    def __init__(self, algorithm_name):
        self.algorithm_name = algorithm_name
        self.reset()

    def reset(self):
        # Separate tracking for different types of gain function calls
        self.gain_function_calls = 0  # g(S) calls - ACTUAL oracle calls only
        self.marginal_gain_calls = 0  # g(e|S) calls (computed as g(S∪{e}) - g(S))
        self.singleton_gain_calls = 0  # g({e}) calls

        # Detailed breakdown of gain calls in marginal computation
        self.gain_union_calls = 0  # g(S ∪ {e}) calls - ACTUAL oracle calls
        self.gain_current_set_calls = 0  # g(S) calls - ACTUAL oracle calls (not cached)

        self.cost_function_calls = 0
        self.set_operations = 0  # Union, intersection, etc.
        self.comparisons = 0
        self.iterations = 0
        self.threshold_checks = 0

        # NEW: Track cached vs actual calls
        self.cached_gain_hits = 0

    def count_gain_call(self, call_type="general"):
        """Count ACTUAL gain function calls (not cached)"""
        self.gain_function_calls += 1
        if call_type == "union":
            self.gain_union_calls += 1
        elif call_type == "current_set":
            self.gain_current_set_calls += 1

    def count_cached_hit(self):
        """Count when we use cached result instead of oracle call"""
        self.cached_gain_hits += 1

    def count_marginal_gain_call(self):
        """Count when we compute a marginal gain g(e|S) - NO oracle calls here"""
        self.marginal_gain_calls += 1

    def count_singleton_gain_call(self):
        """Count when we compute g({e})"""
        self.singleton_gain_calls += 1

    def count_cost_call(self):
        self.cost_function_calls += 1

    def count_set_operation(self):
        self.set_operations += 1

    def count_comparison(self):
        self.comparisons += 1

    def count_iteration(self):
        self.iterations += 1

    def count_threshold_check(self):
        self.threshold_checks += 1

    def get_summary(self):
        return {
            'algorithm': self.algorithm_name,
            'total_gain_calls': self.gain_function_calls,
            'marginal_gain_calls': self.marginal_gain_calls,
            'singleton_gain_calls': self.singleton_gain_calls,
            'gain_union_calls': self.gain_union_calls,
            'gain_current_set_calls': self.gain_current_set_calls,
            'cached_hits': self.cached_gain_hits,
            'cost_calls': self.cost_function_calls,
            'set_operations': self.set_operations,
            'comparisons': self.comparisons,
            'iterations': self.iterations,
            'threshold_checks': self.threshold_checks,
            'total_oracle_calls': self.gain_function_calls + self.singleton_gain_calls
        }

    def print_summary(self):
        summary = self.get_summary()
        print(f"\n=== {self.algorithm_name} CORRECTED OPERATION SUMMARY ===")
        print(f"ACTUAL Oracle calls: {summary['total_oracle_calls']}")
        print(f"  ├── g(S ∪ {{e}}) calls: {summary['gain_union_calls']}")
        print(f"  ├── g(S) calls: {summary['gain_current_set_calls']}")
        print(f"  ├── g({{e}}) calls: {summary['singleton_gain_calls']}")
        print(
            f"  └── Other g(·) calls: {summary['total_gain_calls'] - summary['gain_union_calls'] - summary['gain_current_set_calls']}")
        print(f"Cached hits (no oracle): {summary['cached_hits']}")
        print(f"Marginal gain g(e|S) computations: {summary['marginal_gain_calls']}")
        print(f"Cost function calls: {summary['cost_calls']}")
        print(f"Set operations (∪, ∩, \\): {summary['set_operations']}")
        print(f"Comparisons: {summary['comparisons']}")
        print(f"Iterations: {summary['iterations']}")
        print(f"Threshold checks: {summary['threshold_checks']}")


class GainCache:
    """Smart caching for gain function to minimize oracle calls"""

    def __init__(self, gain_fn, tracker):
        self.gain_fn = gain_fn
        self.tracker = tracker
        self.cache = {}

    def get_gain(self, region_list, call_type="general"):
        """Get gain with caching - only count oracle if not cached"""
        if not region_list:
            return 0

        # Create hashable key from region IDs
        key = tuple(sorted([r['id'] for r in region_list]))

        if key in self.cache:
            self.tracker.count_cached_hit()
            return self.cache[key]
        else:
            # This is an ACTUAL oracle call
            self.tracker.count_gain_call(call_type)
            result = self.gain_fn(region_list)
            self.cache[key] = result
            return result


def Greedy_Search_Tracked(images, saliency_maps, N, m, budget, cost_fn, gain_fn):
    """
    CORRECTED Greedy Search: Tính g(S) chỉ 1 lần per iteration
    """
    tracker = OperationTracker("Greedy Search (CORRECTED)")
    print("Greedy: Starting CORRECTED Algorithm GS with proper oracle counting")

    # Line 1: V ← ID(I, N, A, m)
    V = image_division(images, saliency_maps, N, m)
    n = len(V)
    print(f"Greedy: Generated {n} subregions")

    # Line 2: S ← ∅
    S = []

    # Line 3: U ← V
    U = list(V)
    tracker.count_set_operation()  # Copy operation

    outer_iteration = 0

    # Use caching for efficiency but count oracle calls correctly
    cache = GainCache(gain_fn, tracker)

    # Line 4: repeat
    while True:
        tracker.count_iteration()
        outer_iteration += 1

        print(f"Greedy: Iteration {outer_iteration}, |U|={len(U)}, |S|={len(S)}")

        # CORRECTED: Tính g(S) CHỈ MỘT LẦN mỗi iteration
        current_gain_S = cache.get_gain(S, "current_set")  # 1 oracle call if not cached

        # Line 5: I_t^M ← arg max_{I^M ∈ U} g(I^M|S)/c(I^M)
        best_region = None
        best_density = -1

        # Duyệt tất cả regions trong U
        for I_M in U:  # |U| regions
            # CORRECTED: Chỉ tính g(S ∪ {I^M}), không tính lại g(S)
            gain_union = cache.get_gain(S + [I_M], "union")  # 1 oracle call per region

            # g(I^M|S) = g(S ∪ {I^M}) - g(S) - KHÔNG CẦN oracle call thêm
            tracker.count_marginal_gain_call()
            marginal_g = gain_union - current_gain_S

            # Calculate c(I^M)
            tracker.count_cost_call()
            cost = cost_fn(I_M)

            if cost > 0:
                density = marginal_g / cost
                tracker.count_comparison()
                if density > best_density:
                    best_density = density
                    best_region = I_M

        if best_region is None:
            print("Greedy: No more valid regions found")
            break

        # Line 6: if c(S + I_t^M) ≤ B then
        # Calculate current cost c(S)
        current_cost = 0
        for x in S:
            tracker.count_cost_call()
            current_cost += cost_fn(x)

        # Calculate c(I_t^M)
        tracker.count_cost_call()
        new_region_cost = cost_fn(best_region)

        # Check budget constraint
        tracker.count_comparison()
        if current_cost + new_region_cost <= budget:
            # Line 7: S = S + I_t^M
            tracker.count_set_operation()  # Union operation
            S.append(best_region)
            print(f"Greedy: Added region {best_region['id']}, density={best_density:.3f}")
        else:
            print("Greedy: Budget constraint violated, stopping")
            break

        # Line 8: U ← U \ {I_t^M}
        tracker.count_set_operation()  # Set difference operation
        U.remove(best_region)

        # Line 9: until U = ∅
        if not U:
            print("Greedy: All regions processed")
            break

    # Final gain calculation
    total_gain = cache.get_gain(S) if S else 0

    print(f"Greedy: CORRECTED complexity analysis:")
    print(f"  Expected oracle calls: |S| + Σ|U_i| = {len(S)} + {sum(len(U) - i for i in range(len(S)))}")

    tracker.print_summary()

    # Memory aux data (giữ nguyên format cũ)
    memory_aux = {'operation_summary': tracker.get_summary()}

    return (S, total_gain), memory_aux


def OT_Algorithm_Tracked(images, saliency_maps, N, m, budget, cost_fn, gain_fn):
    """
    CORRECTED OT Algorithm: Sử dụng caching để đạt chính xác 3n oracle calls
    """
    tracker = OperationTracker("Online Threshold (CORRECTED)")
    print("OT: Starting CORRECTED Algorithm 2 with proper oracle counting")

    V = image_division(images, saliency_maps, N, m)
    n = len(V)

    # Line 1: V, S, S', I* ← ∅
    S, S_prime, I_star = [], [], None
    I_star_gain = 0

    # Use smart caching
    cache = GainCache(gain_fn, tracker)

    # Line 2: for an incoming image I do (streaming simulation)
    for I_M in V:
        tracker.count_iteration()

        # Skip infeasible regions early (optimization)
        tracker.count_cost_call()
        if cost_fn(I_M) > budget:
            continue

        # Line 6: S_d ← arg max_{S_d ∈ {S,S'}} g(I^M|S_d)
        # CORRECTED: Chỉ tính g(S∪{I^M}) và g(S'∪{I^M}), cache g(S) và g(S')

        gain_S_union = cache.get_gain(S + [I_M], "union")  # 1 oracle call
        gain_S = cache.get_gain(S, "current_set")  # Oracle call hoặc cache hit

        gain_S_prime_union = cache.get_gain(S_prime + [I_M], "union")  # 1 oracle call
        gain_S_prime = cache.get_gain(S_prime, "current_set")  # Oracle call hoặc cache hit

        # Marginal gains - KHÔNG CẦN oracle calls
        tracker.count_marginal_gain_call()
        marginal_S = gain_S_union - gain_S

        tracker.count_marginal_gain_call()
        marginal_S_prime = gain_S_prime_union - gain_S_prime

        tracker.count_comparison()
        if marginal_S >= marginal_S_prime:
            S_d = S
            marginal_g = marginal_S
            current_gain = gain_S
            is_S = True
        else:
            S_d = S_prime
            marginal_g = marginal_S_prime
            current_gain = gain_S_prime
            is_S = False

        # Line 7: if g(I^M|S_d)/c(I^M) ≥ g(S_d)/B then
        tracker.count_cost_call()
        region_cost = cost_fn(I_M)

        tracker.count_threshold_check()
        tracker.count_comparison()
        if region_cost > 0 and marginal_g / region_cost >= current_gain / budget:
            # Line 8: S_d = S_d + I^M
            tracker.count_set_operation()  # Union operation
            if is_S:
                S.append(I_M)
            else:
                S_prime.append(I_M)

        # Line 9: I* = arg max_{I* ∈ {I*,I^M}} g(I*)
        tracker.count_singleton_gain_call()
        singleton_gain = gain_fn([I_M])  # 1 oracle call - không thể cache singleton

        tracker.count_comparison()
        if I_star is None or singleton_gain > I_star_gain:
            I_star = I_M
            I_star_gain = singleton_gain

    # Lines 10-19: Final selection (giữ nguyên logic cũ)
    # Calculate costs
    S_cost = sum(cost_fn(x) for x in S)
    S_prime_cost = sum(cost_fn(x) for x in S_prime)
    I_star_cost = cost_fn(I_star) if I_star else float('inf')

    for x in S:
        tracker.count_cost_call()
    for x in S_prime:
        tracker.count_cost_call()
    if I_star:
        tracker.count_cost_call()

    # Calculate final gains for decision making
    gain_S = cache.get_gain(S) if S else 0
    gain_S_prime = cache.get_gain(S_prime) if S_prime else 0

    # Final selection logic (Lines 10-18) - giữ nguyên
    candidates = []

    tracker.count_comparison()
    tracker.count_comparison()
    if S_cost <= budget and S_prime_cost <= budget:
        # Line 11
        candidates = [(S, gain_S), (S_prime, gain_S_prime)]
        if I_star_cost <= budget:
            candidates.append(([I_star], I_star_gain))
        tracker.count_comparison()
        S_star, g_star = max(candidates, key=lambda x: x[1])
    else:
        # Handle infeasible cases (simplified)
        if S_cost <= budget:
            S_star, g_star = S, gain_S
        elif S_prime_cost <= budget:
            S_star, g_star = S_prime, gain_S_prime
        elif I_star_cost <= budget:
            S_star, g_star = [I_star], I_star_gain
        else:
            S_star, g_star = [], 0

    print(f"OT: CORRECTED - Expected ~3n = {3 * n} oracle calls")
    tracker.print_summary()

    memory_aux = {
        'S_size': len(S),
        'S_prime_size': len(S_prime),
        'has_I_star': I_star is not None,
        'operation_summary': tracker.get_summary()
    }

    return (S_star, g_star), memory_aux


def IOT_Algorithm_Tracked(images, saliency_maps, N, m, budget, eps, cost_fn, gain_fn):
    """
    CORRECTED IOT Algorithm: Sử dụng caching để đạt chính xác 5n oracle calls
    """
    tracker = OperationTracker("Improved Online Threshold (CORRECTED)")
    print("IOT: Starting CORRECTED Algorithm 3 with proper oracle counting")

    # Line 2: ε' = ε/5
    eps_prime = eps / 5

    # Line 3: S_b, M ← OT(N, A, m, B) - First stream
    (S_b, M), ot_memory = OT_Algorithm_Tracked(images, saliency_maps, N, m, budget, cost_fn, gain_fn)

    # Add OT operations to IOT tracker
    ot_summary = ot_memory['operation_summary']
    tracker.gain_function_calls += ot_summary.get('total_gain_calls', 0)
    tracker.marginal_gain_calls += ot_summary.get('marginal_gain_calls', 0)
    tracker.singleton_gain_calls += ot_summary.get('singleton_gain_calls', 0)
    tracker.gain_union_calls += ot_summary.get('gain_union_calls', 0)
    tracker.gain_current_set_calls += ot_summary.get('gain_current_set_calls', 0)
    tracker.cached_gain_hits += ot_summary.get('cached_hits', 0)
    tracker.cost_function_calls += ot_summary.get('cost_calls', 0)
    tracker.set_operations += ot_summary.get('set_operations', 0)
    tracker.comparisons += ot_summary.get('comparisons', 0)
    tracker.threshold_checks += ot_summary.get('threshold_checks', 0)

    # Line 4: Generate threshold set T (giữ nguyên logic)
    T = []
    lower_bound = (1 - eps_prime) * eps_prime * M / (2 * budget)
    upper_bound = 4 * M / (eps_prime * budget)

    num_thresholds = min(10, max(3, int(5 / eps_prime)))
    if lower_bound < upper_bound:
        ratio = (upper_bound / lower_bound) ** (1.0 / (num_thresholds - 1))
        tau = lower_bound
        for i in range(num_thresholds):
            if tau > upper_bound:
                break
            T.append(tau)
            tau *= ratio
    else:
        T = [lower_bound]

    print(f"IOT: Using {len(T)} thresholds")

    # Initialize candidates for each threshold
    candidates = {}
    for tau in T:
        candidates[tau] = {'S_tau': [], 'S_prime_tau': []}

    # Use smart caching for second pass
    cache = GainCache(gain_fn, tracker)

    # Line 5-7: Second stream
    V = image_division(images, saliency_maps, N, m)

    for I_M in V:
        tracker.count_iteration()

        # Line 8-12: For each threshold (giữ nguyên logic, sửa oracle counting)
        for tau in T:
            S_tau = candidates[tau]['S_tau']
            S_prime_tau = candidates[tau]['S_prime_tau']

            # CORRECTED: Chỉ tính union, cache current sets
            gain_S_tau_union = cache.get_gain(S_tau + [I_M], "union")  # 1 oracle call
            gain_S_tau = cache.get_gain(S_tau, "current_set")  # Oracle or cache hit

            gain_S_prime_tau_union = cache.get_gain(S_prime_tau + [I_M], "union")  # 1 oracle call
            gain_S_prime_tau = cache.get_gain(S_prime_tau, "current_set")  # Oracle or cache hit

            # Marginal gains - NO oracle calls
            tracker.count_marginal_gain_call()
            marginal_S_tau = gain_S_tau_union - gain_S_tau

            tracker.count_marginal_gain_call()
            marginal_S_prime_tau = gain_S_prime_tau_union - gain_S_prime_tau

            tracker.count_comparison()
            if marginal_S_tau >= marginal_S_prime_tau:
                X_tau = S_tau
                target_key = 'S_tau'
                marginal_g = marginal_S_tau
            else:
                X_tau = S_prime_tau
                target_key = 'S_prime_tau'
                marginal_g = marginal_S_prime_tau

            # Line 11: Threshold and budget conditions
            tracker.count_cost_call()
            region_cost = cost_fn(I_M)

            current_cost = sum(cost_fn(x) for x in X_tau)
            for x in X_tau:
                tracker.count_cost_call()

            tracker.count_threshold_check()
            tracker.count_comparison()
            tracker.count_comparison()
            if (region_cost > 0 and
                    marginal_g / region_cost >= tau and
                    current_cost + region_cost <= budget):
                # Line 12: X_τ = X_τ + I^M
                tracker.count_set_operation()
                candidates[tau][target_key] = X_tau + [I_M]

    # Line 13-15: Final selection (giữ nguyên)
    all_candidates = [S_b]

    max_S_tau_size = 0
    max_S_prime_tau_size = 0

    for tau in T:
        S_tau = candidates[tau]['S_tau']
        S_prime_tau = candidates[tau]['S_prime_tau']

        max_S_tau_size = max(max_S_tau_size, len(S_tau))
        max_S_prime_tau_size = max(max_S_prime_tau_size, len(S_prime_tau))

        # Check feasibility
        S_tau_cost = sum(cost_fn(x) for x in S_tau)
        S_prime_tau_cost = sum(cost_fn(x) for x in S_prime_tau)

        for x in S_tau:
            tracker.count_cost_call()
        for x in S_prime_tau:
            tracker.count_cost_call()

        tracker.count_comparison()
        if S_tau_cost <= budget:
            all_candidates.append(S_tau)
        tracker.count_comparison()
        if S_prime_tau_cost <= budget:
            all_candidates.append(S_prime_tau)

    # Evaluate candidates
    if not all_candidates:
        S_star, g_star = [], 0
    else:
        candidate_gains = []
        for candidate in all_candidates:
            gain = cache.get_gain(candidate) if candidate else 0
            candidate_gains.append((candidate, gain))

        tracker.count_comparison()
        S_star, g_star = max(candidate_gains, key=lambda x: x[1])

    n = len(V)
    print(f"IOT: CORRECTED - Expected ~5n = {5 * n} oracle calls")
    tracker.print_summary()

    memory_aux = {
        'S_size': ot_memory['S_size'],
        'S_prime_size': ot_memory['S_prime_size'],
        'has_I_star': ot_memory['has_I_star'],
        'max_S_tau_size': max_S_tau_size,
        'max_S_prime_tau_size': max_S_prime_tau_size,
        'S_b_size': len(S_b),
        'operation_summary': tracker.get_summary()
    }

    return (S_star, g_star), memory_aux