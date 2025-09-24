import numpy as np
from image_division import image_division


class OperationTracker:
    """Track operations and function calls for algorithm analysis"""

    def __init__(self, algorithm_name):
        self.algorithm_name = algorithm_name
        self.reset()

    def reset(self):
        self.gain_function_calls = 0
        self.marginal_gain_calls = 0
        self.cost_function_calls = 0
        self.set_operations = 0  # Union, intersection, etc.
        self.comparisons = 0
        self.iterations = 0
        self.threshold_checks = 0

    def count_gain_call(self):
        self.gain_function_calls += 1

    def count_marginal_gain_call(self):
        self.marginal_gain_calls += 1

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
            'gain_calls': self.gain_function_calls,
            'marginal_gain_calls': self.marginal_gain_calls,
            'cost_calls': self.cost_function_calls,
            'set_operations': self.set_operations,
            'comparisons': self.comparisons,
            'iterations': self.iterations,
            'threshold_checks': self.threshold_checks,
            'total_oracle_calls': self.gain_function_calls + self.marginal_gain_calls
        }

    def print_summary(self):
        summary = self.get_summary()
        print(f"\n=== {self.algorithm_name} OPERATION SUMMARY ===")
        print(f"Gain function calls g(S): {summary['gain_calls']}")
        print(f"Marginal gain calls g(e|S): {summary['marginal_gain_calls']}")
        print(f"Total oracle calls: {summary['total_oracle_calls']}")
        print(f"Cost function calls: {summary['cost_calls']}")
        print(f"Set operations (∪, ∩, \\): {summary['set_operations']}")
        print(f"Comparisons: {summary['comparisons']}")
        print(f"Iterations: {summary['iterations']}")
        print(f"Threshold checks: {summary['threshold_checks']}")


def tracked_marginal_gain(element, current_set, gain_fn, tracker):
    """Calculate marginal gain with tracking: g(I^M|S) = g(S ∪ {I^M}) - g(S)"""
    tracker.count_marginal_gain_call()

    if not current_set:
        tracker.count_gain_call()
        return gain_fn([element])

    # g(S ∪ {I^M})
    tracker.count_set_operation()  # Union operation
    tracker.count_gain_call()
    gain_with = gain_fn(current_set + [element])

    # g(S)
    tracker.count_gain_call()
    gain_without = gain_fn(current_set)

    return gain_with - gain_without


def Greedy_Search_Tracked(images, saliency_maps, N, m, budget, cost_fn, gain_fn):
    """
    Greedy Search following Algorithm GS with operation tracking.
    """
    tracker = OperationTracker("Greedy Search (GS)")
    print("Greedy: Starting Algorithm GS with tracking")

    # Line 1: V ← ID(I, N, A, m)
    V = image_division(images, saliency_maps, N, m)
    n = len(V)
    print(f"Greedy: Generated {n} subregions")

    # Line 2: S ← ∅
    S = []

    # Line 3: U ← V
    U = list(V)
    tracker.count_set_operation()  # Copy operation

    # Line 4: repeat
    while True:
        tracker.count_iteration()

        # Line 5: I_t^M ← arg max_{I^M ∈ U} g(I^M|S + I^M)/c(I^M)
        best_region = None
        best_density = -1

        for I_M in U:  # Iterate through remaining regions
            # Calculate g(I^M|S)
            marginal_g = tracked_marginal_gain(I_M, S, gain_fn, tracker)

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
        else:
            break

        # Line 8: U ← U \ {I_t^M}
        tracker.count_set_operation()  # Set difference operation
        U.remove(best_region)

        # Line 9: until U = ∅
        if not U:
            break

    # Final gain calculation
    tracker.count_gain_call()
    total_gain = gain_fn(S) if S else 0

    tracker.print_summary()

    # Memory aux data
    memory_aux = {'operation_summary': tracker.get_summary()}

    return (S, total_gain), memory_aux


def OT_Algorithm_Tracked(images, saliency_maps, N, m, budget, cost_fn, gain_fn):
    """
    OT Algorithm following Algorithm 2 with operation tracking.
    Compatible with original function signature.
    """
    tracker = OperationTracker("Online Threshold (OT)")
    print("OT: Starting Algorithm 2 with tracking")

    V = image_division(images, saliency_maps, N, m)
    n = len(V)

    # Line 1: V, S, S', I* ← ∅
    S, S_prime, I_star = [], [], None
    I_star_gain = 0

    # Line 2: for an incoming image I do (streaming simulation)
    for I_M in V:
        tracker.count_iteration()

        # Skip infeasible regions early
        tracker.count_cost_call()
        if cost_fn(I_M) > budget:
            continue

        # Line 5: for each region I^M ∈ V \ {S + S'} do
        tracker.count_set_operation()  # Check membership S + S'
        if I_M in S or I_M in S_prime:
            continue

        # Line 6: S_d ← arg max_{S_d ∈ {S,S'}} g(I^M|S_d)
        marginal_S = tracked_marginal_gain(I_M, S, gain_fn, tracker)
        marginal_S_prime = tracked_marginal_gain(I_M, S_prime, gain_fn, tracker)

        tracker.count_comparison()
        if marginal_S >= marginal_S_prime:
            S_d = S
            marginal_g = marginal_S
            is_S = True
        else:
            S_d = S_prime
            marginal_g = marginal_S_prime
            is_S = False

        # Line 7: if g(I^M|S_d)/c(I^M) ≥ g(S_d)/B then
        tracker.count_cost_call()
        region_cost = cost_fn(I_M)

        tracker.count_gain_call()
        current_gain = gain_fn(S_d)

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
        tracker.count_gain_call()
        singleton_gain = gain_fn([I_M])

        tracker.count_comparison()
        if I_star is None or singleton_gain > I_star_gain:
            I_star = I_M
            I_star_gain = singleton_gain

    # Lines 10-19: Final selection
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

    # Calculate final gains
    tracker.count_gain_call()
    gain_S = gain_fn(S) if S else 0
    tracker.count_gain_call()
    gain_S_prime = gain_fn(S_prime) if S_prime else 0

    # Final selection logic (Lines 10-18)
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
        # Handle infeasible cases (simplified for brevity)
        if S_cost <= budget:
            S_star, g_star = S, gain_S
        elif S_prime_cost <= budget:
            S_star, g_star = S_prime, gain_S_prime
        elif I_star_cost <= budget:
            S_star, g_star = [I_star], I_star_gain
        else:
            S_star, g_star = [], 0

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
    IOT Algorithm following Algorithm 3 with operation tracking.
    """
    tracker = OperationTracker("Improved Online Threshold (IOT)")
    print("IOT: Starting Algorithm 3 with tracking")

    # Line 2: ε' = ε/5
    eps_prime = eps / 5

    # Line 3: S_b, M ← OT(N, A, m, B) - First stream
    (S_b, M), ot_memory = OT_Algorithm_Tracked(images, saliency_maps, N, m, budget, cost_fn, gain_fn)

    # Add OT operations to IOT tracker
    ot_summary = ot_memory['operation_summary']
    tracker.gain_function_calls += ot_summary['gain_calls']
    tracker.marginal_gain_calls += ot_summary['marginal_gain_calls']
    tracker.cost_function_calls += ot_summary['cost_calls']
    tracker.set_operations += ot_summary['set_operations']
    tracker.comparisons += ot_summary['comparisons']
    tracker.threshold_checks += ot_summary['threshold_checks']

    # Line 4: Generate threshold set T
    T = []
    lower_bound = (1 - eps_prime) * eps_prime * M / (2 * budget)
    upper_bound = 4 * M / (eps_prime * budget)

    # Generate thresholds (simplified)
    num_thresholds = min(10, max(3, int(5 / eps_prime)))  # Reasonable number
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

    # Line 5-7: Second stream
    V = image_division(images, saliency_maps, N, m)

    for I_M in V:
        tracker.count_iteration()

        # Line 8: for each unconsidered region I^M ∈ V do
        # Line 9: for each τ ∈ T do
        for tau in T:
            S_tau = candidates[tau]['S_tau']
            S_prime_tau = candidates[tau]['S_prime_tau']

            # Line 10: X_τ ← arg max_{X_τ ∈ {S_τ,S'_τ}} g(I^M|X_τ)
            marginal_S_tau = tracked_marginal_gain(I_M, S_tau, gain_fn, tracker)
            marginal_S_prime_tau = tracked_marginal_gain(I_M, S_prime_tau, gain_fn, tracker)

            tracker.count_comparison()
            if marginal_S_tau >= marginal_S_prime_tau:
                X_tau = S_tau
                target_key = 'S_tau'
                marginal_g = marginal_S_tau
            else:
                X_tau = S_prime_tau
                target_key = 'S_prime_tau'
                marginal_g = marginal_S_prime_tau

            # Line 11: if g(I^M|X_τ)/c(I^M) ≥ τ ∧ c(X_τ) ≤ B then
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

    # Line 13-15: Final selection
    all_candidates = [S_b]

    for tau in T:
        S_tau = candidates[tau]['S_tau']
        S_prime_tau = candidates[tau]['S_prime_tau']

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
            tracker.count_gain_call()
            gain = gain_fn(candidate) if candidate else 0
            candidate_gains.append((candidate, gain))

        tracker.count_comparison()
        S_star, g_star = max(candidate_gains, key=lambda x: x[1])

    tracker.print_summary()

    memory_aux = {
        'S_size': ot_memory['S_size'],
        'S_prime_size': ot_memory['S_prime_size'],
        'has_I_star': ot_memory['has_I_star'],
        'max_S_tau_size': max(len(candidates[tau]['S_tau']) for tau in T),
        'max_S_prime_tau_size': max(len(candidates[tau]['S_prime_tau']) for tau in T),
        'S_b_size': len(S_b),
        'operation_summary': tracker.get_summary()
    }

    return (S_star, g_star), memory_aux