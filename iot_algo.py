import math
from ot_algo import OT_Algorithm_Corrected, OT_algorithm
from image_division import image_division
from tracked_algorithms import OperationTracker


def IOT_Algorithm_Corrected(images, saliency_maps, N, m, budget, eps, cost_fn, gain_fn):
    """
    Corrected IOT algorithm with proper caching and oracle call tracking
    """
    tracker = OperationTracker("IOT (Corrected)")

    # Line 3: First pass with OT (corrected version)
    (Sb, M), _ = OT_Algorithm_Corrected(images, saliency_maps, N, m, budget, cost_fn, gain_fn)
    M = M if M > 0 else 1  # Avoid division by zero

    # Line 2: Set epsilon prime
    eps_prime = eps / 5

    # Line 4: Generate threshold set T
    T = []
    lower_bound = max(1e-6, (1 - eps_prime) * eps_prime * M / (2 * budget))
    upper_bound = min(M, 4 * M / (eps_prime * budget))

    if lower_bound >= upper_bound:
        T = [lower_bound]
    else:
        num_thresholds = max(5, int(10 / eps_prime))
        ratio = (upper_bound / lower_bound) ** (1.0 / (num_thresholds - 1))
        tau = lower_bound
        for i in range(num_thresholds):
            if tau > upper_bound:
                break
            T.append(tau)
            tau *= ratio

    # Cache để tránh tính lại
    cached_gains = {}

    def get_cached_gain(region_list):
        if not region_list:
            return 0
        key = tuple(sorted([r['id'] for r in region_list]))
        if key not in cached_gains:
            tracker.count_gain_call()
            cached_gains[key] = gain_fn(region_list)
        return cached_gains[key]

    # Initialize candidate sets for each threshold
    candidates = {}
    for tau in T:
        candidates[tau] = {'S_tau': [], 'S_prime_tau': []}

    # Line 5-7: Second pass through data stream
    V = image_division(images, saliency_maps, N, m)

    for I_M in V:
        tracker.count_iteration()

        # Line 8-12: For each threshold
        for tau in T:
            S_tau = candidates[tau]['S_tau']
            S_prime_tau = candidates[tau]['S_prime_tau']

            # Early termination if region cost too high
            if cost_fn(I_M) > budget:
                continue

            # Tính marginal gains với cache
            gain_S_tau_union = get_cached_gain(S_tau + [I_M])
            gain_S_prime_tau_union = get_cached_gain(S_prime_tau + [I_M])

            gain_S_tau = get_cached_gain(S_tau)
            gain_S_prime_tau = get_cached_gain(S_prime_tau)

            marginal_S_tau = gain_S_tau_union - gain_S_tau
            marginal_S_prime_tau = gain_S_prime_tau_union - gain_S_prime_tau

            # Choose candidate with higher marginal gain
            if marginal_S_tau >= marginal_S_prime_tau:
                X_tau = S_tau
                target_key = 'S_tau'
                marginal_g = marginal_S_tau
            else:
                X_tau = S_prime_tau
                target_key = 'S_prime_tau'
                marginal_g = marginal_S_prime_tau

            # Line 11: Threshold condition and budget constraint
            current_cost = sum(cost_fn(x) for x in X_tau)

            if (marginal_g / cost_fn(I_M) >= tau and
                    current_cost + cost_fn(I_M) <= budget):
                # Line 12: Add element to selected candidate
                candidates[tau][target_key] = X_tau + [I_M]

    # Line 13: Find best solution among all candidates
    all_candidates = [Sb]  # Include first pass solution

    for tau in T:
        S_tau = candidates[tau]['S_tau']
        S_prime_tau = candidates[tau]['S_prime_tau']

        if sum(cost_fn(x) for x in S_tau) <= budget:
            all_candidates.append(S_tau)
        if sum(cost_fn(x) for x in S_prime_tau) <= budget:
            all_candidates.append(S_prime_tau)

    # Line 14: Return best solution
    if not all_candidates:
        return ([], 0)

    S_star = max(all_candidates, key=lambda X: get_cached_gain(X) if X else 0)
    final_gain = get_cached_gain(S_star) if S_star else 0

    return (S_star, final_gain)


def marginal_gain(element, current_set, gain_fn):
    """Calculate marginal gain: g(element|current_set)"""
    gain_with = gain_fn(current_set + [element])
    gain_without = gain_fn(current_set) if current_set else 0
    return gain_with - gain_without


def IOT_algorithm(images, saliency_maps, N, m, budget, eps, cost_fn, gain_fn):
    """
    Optimized IOT algorithm with memory tracking.
    Returns: ((solution_set, gain), memory_aux_data)
    """
    print(f"IOT: Starting with ε={eps}")

    # Line 3: First pass with OT
    (Sb, M), ot_memory = OT_algorithm(images, saliency_maps, N, m, budget, cost_fn, gain_fn)
    M = M if M > 0 else 1  # Avoid division by zero
    print(f"IOT: OT baseline gain = {M}")

    # Line 2: Set epsilon prime
    eps_prime = eps / 5

    # Line 4: Generate SMALLER threshold set T (optimization)
    T = []
    # More aggressive threshold reduction
    min_thresholds = 5  # Minimum number of thresholds
    max_thresholds = max(min_thresholds, int(10 / eps_prime))  # Cap at reasonable size

    lower_bound = max(1e-6, (1 - eps_prime) * eps_prime * M / (2 * budget))
    upper_bound = min(M, 4 * M / (eps_prime * budget))

    if lower_bound >= upper_bound:
        T = [lower_bound]
    else:
        # Geometric progression with fewer steps
        ratio = (upper_bound / lower_bound) ** (1.0 / (max_thresholds - 1))
        tau = lower_bound
        for i in range(max_thresholds):
            if tau > upper_bound:
                break
            T.append(tau)
            tau *= ratio

    print(f"IOT: Using {len(T)} thresholds: {[f'{t:.2e}' for t in T[:3]]}...")

    # Initialize candidate sets for each threshold
    candidates = {}
    for tau in T:
        candidates[tau] = {'S_tau': [], 'S_prime_tau': []}

    # Line 5-7: Second pass through data stream
    V = image_division(images, saliency_maps, N, m)
    print(f"IOT: Processing {len(V)} regions")

    # Cache for expensive computations
    gain_cache = {}

    def cached_gain(regions_list):
        """Cache gain function calls using region IDs as key"""
        if not regions_list:
            return 0
        # Create hashable key from region IDs
        key = tuple(sorted([r['id'] for r in regions_list]))
        if key not in gain_cache:
            gain_cache[key] = gain_fn(regions_list)
        return gain_cache[key]

    processed = 0
    for I_M in V:
        processed += 1
        if processed % 10 == 0:
            print(f"IOT: Processed {processed}/{len(V)} regions")

        # Line 8-12: For each threshold (optimized)
        for tau_idx, tau in enumerate(T):
            S_tau = candidates[tau]['S_tau']
            S_prime_tau = candidates[tau]['S_prime_tau']

            # Early termination if region cost too high
            if cost_fn(I_M) > budget:
                continue

            # Choose candidate with higher marginal gain (using cache)
            gain_S_tau = 0 if not S_tau else (
                    cached_gain(S_tau + [I_M]) - cached_gain(S_tau)
            )
            gain_S_prime_tau = 0 if not S_prime_tau else (
                    cached_gain(S_prime_tau + [I_M]) - cached_gain(S_prime_tau)
            )

            if gain_S_tau >= gain_S_prime_tau:
                X_tau = S_tau
                target_key = 'S_tau'
                marginal_g = gain_S_tau
            else:
                X_tau = S_prime_tau
                target_key = 'S_prime_tau'
                marginal_g = gain_S_prime_tau

            # Line 11: Threshold condition and budget constraint
            current_cost = sum(cost_fn(x) for x in X_tau)

            if (marginal_g / cost_fn(I_M) >= tau and
                    current_cost + cost_fn(I_M) <= budget):
                # Line 12: Add element to selected candidate
                candidates[tau][target_key] = X_tau + [I_M]

    print("IOT: Finding best solution...")

    # Line 13: Find best solution among all candidates
    all_candidates = [Sb]  # Include first pass solution

    # Track sizes for memory calculation
    max_S_tau_size = 0
    max_S_prime_tau_size = 0

    for tau in T:
        S_tau = candidates[tau]['S_tau']
        S_prime_tau = candidates[tau]['S_prime_tau']

        max_S_tau_size = max(max_S_tau_size, len(S_tau))
        max_S_prime_tau_size = max(max_S_prime_tau_size, len(S_prime_tau))

        if sum(cost_fn(x) for x in S_tau) <= budget:
            all_candidates.append(S_tau)
        if sum(cost_fn(x) for x in S_prime_tau) <= budget:
            all_candidates.append(S_prime_tau)

    # Line 14: Return best solution
    if not all_candidates:
        memory_aux = {}
        return ([], 0), memory_aux

    print("IOT: Evaluating final candidates...")
    S_star = max(all_candidates, key=lambda X: cached_gain(X) if X else 0)
    final_gain = cached_gain(S_star) if S_star else 0

    print(f"IOT: Completed. Final gain = {final_gain}, |S*| = {len(S_star)}")

    # Memory aux data for IOT:
    # M = m * sizeof(1 sub) + |S+S'+1| * sizeof(1 sub) + |max of S_tau + max of S'_tau + S_b| * sizeof(1 sub)
    memory_aux = {
        'S_size': ot_memory.get('S_size', 0),
        'S_prime_size': ot_memory.get('S_prime_size', 0),
        'has_I_star': ot_memory.get('has_I_star', False),
        'max_S_tau_size': max_S_tau_size,
        'max_S_prime_tau_size': max_S_prime_tau_size,
        'S_b_size': len(Sb)
    }

    return (S_star, final_gain), memory_aux