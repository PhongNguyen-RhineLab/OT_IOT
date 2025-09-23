import math
from ot_algo import OT_algorithm
from image_division import image_division


def marginal_gain(element, current_set, gain_fn):
    """Calculate marginal gain: g(element|current_set)"""
    gain_with = gain_fn(current_set + [element])
    gain_without = gain_fn(current_set) if current_set else 0
    return gain_with - gain_without


def IOT_algorithm_optimized(images, saliency_maps, N, m, budget, eps, cost_fn, gain_fn):
    """
    Optimized IOT algorithm with:
    1. Reduced threshold set size
    2. Early termination
    3. Cached computations
    """
    print(f"IOT: Starting with ε={eps}")

    # Line 3: First pass with OT
    Sb, M = OT_algorithm(images, saliency_maps, N, m, budget, cost_fn, gain_fn)
    M = gain_fn(Sb) if Sb else 1  # Avoid division by zero
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

    def cached_gain(regions_tuple):
        if regions_tuple not in gain_cache:
            regions_list = list(regions_tuple) if regions_tuple else []
            gain_cache[regions_tuple] = gain_fn(regions_list)
        return gain_cache[regions_tuple]

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
            S_tau_tuple = tuple(f"{r['id']}" for r in S_tau)
            S_prime_tau_tuple = tuple(f"{r['id']}" for r in S_prime_tau)

            gain_S_tau = 0 if not S_tau else (
                    cached_gain(S_tau_tuple + (I_M['id'],)) - cached_gain(S_tau_tuple)
            )
            gain_S_prime_tau = 0 if not S_prime_tau else (
                    cached_gain(S_prime_tau_tuple + (I_M['id'],)) - cached_gain(S_prime_tau_tuple)
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

    for tau in T:
        S_tau = candidates[tau]['S_tau']
        S_prime_tau = candidates[tau]['S_prime_tau']

        if sum(cost_fn(x) for x in S_tau) <= budget:
            all_candidates.append(S_tau)
        if sum(cost_fn(x) for x in S_prime_tau) <= budget:
            all_candidates.append(S_prime_tau)

    # Line 14: Return best solution
    if not all_candidates:
        return [], 0

    print("IOT: Evaluating final candidates...")
    S_star = max(all_candidates, key=lambda X: gain_fn(X) if X else 0)
    final_gain = gain_fn(S_star) if S_star else 0

    print(f"IOT: Completed. Final gain = {final_gain}, |S*| = {len(S_star)}")
    return S_star, final_gain


# Alias for backward compatibility
IOT_algorithm = IOT_algorithm_optimized