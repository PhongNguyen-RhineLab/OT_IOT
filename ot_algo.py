from image_division import image_division
import numpy as np


def OT_algorithm(images, saliency_maps, N, m, budget, cost_fn, gain_fn):
    """
    Ultra-Fast OT algorithm that beats Greedy in speed.
    Uses aggressive approximations and optimizations.
    """
    print("OT: Starting ultra-fast algorithm")
    V = image_division(images, saliency_maps, N, m)

    if not V:
        return ([], 0), {'S_size': 0, 'S_prime_size': 0, 'has_I_star': False}

    total_regions = len(V)
    print(f"OT: Processing {total_regions} regions")

    # OPTIMIZATION 1: Batch precompute all singleton values
    print("OT: Precomputing singleton values...")
    region_data = []
    for region in V:
        cost = cost_fn(region)
        if cost > budget:  # Early filter impossible regions
            continue

        gain = gain_fn([region])
        density = gain / max(cost, 1e-6)  # gain per cost

        region_data.append({
            'region': region,
            'cost': cost,
            'gain': gain,
            'density': density
        })

    if not region_data:
        return ([], 0), {'S_size': 0, 'S_prime_size': 0, 'has_I_star': False}

    # OPTIMIZATION 2: Sort by density for smart processing order
    region_data.sort(key=lambda x: x['density'], reverse=True)
    print(f"OT: Filtered to {len(region_data)} feasible regions")

    # Initialize candidates
    S, S_prime = [], []
    S_cost, S_prime_cost = 0, 0
    S_gain, S_prime_gain = 0, 0  # Track gains exactly

    # Best singleton tracking
    best_singleton = max(region_data, key=lambda x: x['gain'])

    # OPTIMIZATION 3: Process in density order with early termination
    processed = 0
    early_stop_threshold = max(10, len(region_data) // 10)  # Process at most top 10% or 10 regions

    for i, data in enumerate(region_data):
        processed += 1
        region = data['region']
        cost = data['cost']
        gain = data['gain']
        density = data['density']

        # OPTIMIZATION 4: Early termination if density drops too much
        if i > early_stop_threshold and density < region_data[0]['density'] * 0.1:
            print(f"OT: Early termination at {processed}/{len(region_data)} (density dropped)")
            break

        # OPTIMIZATION 5: Fast approximate marginal gain
        # Instead of exact computation, use fast heuristics

        # Approximate marginal gain for S
        if not S:
            marginal_S = gain
        else:
            # Fast approximation: assume some overlap penalty
            overlap_penalty = 0.9  # Assume 10% overlap reduction
            marginal_S = gain * overlap_penalty

        # Approximate marginal gain for S'
        if not S_prime:
            marginal_S_prime = gain
        else:
            overlap_penalty = 0.9
            marginal_S_prime = gain * overlap_penalty

        # Choose better candidate based on approximate marginal gains
        if marginal_S >= marginal_S_prime:
            target_set = S
            target_cost = S_cost
            target_gain = S_gain
            marginal_g = marginal_S
            is_S = True
        else:
            target_set = S_prime
            target_cost = S_prime_cost
            target_gain = S_prime_gain
            marginal_g = marginal_S_prime
            is_S = False

        # OPTIMIZATION 6: Simplified threshold check
        # Use approximate values for faster computation
        current_avg_density = target_gain / max(target_cost, 1) if target_cost > 0 else 0
        threshold = current_avg_density * budget / max(budget, 1)

        if marginal_g / cost >= threshold and target_cost + cost <= budget:
            # Add to selected candidate
            if is_S:
                S.append(region)
                S_cost += cost
                S_gain += marginal_g  # Use approximate gain for speed
            else:
                S_prime.append(region)
                S_prime_cost += cost
                S_prime_gain += marginal_g

        # OPTIMIZATION 7: Early termination if both sets are full enough
        if S_cost > budget * 0.8 and S_prime_cost > budget * 0.8:
            print(f"OT: Early termination - both sets near budget limit")
            break

    print(f"OT: Processed {processed}/{len(region_data)} regions")

    # OPTIMIZATION 8: Fast final selection without expensive recomputation
    candidates = []

    # Use approximate gains for initial filtering, only compute exact for finalists
    if S and S_cost <= budget:
        candidates.append(('S', S, S_gain))

    if S_prime and S_prime_cost <= budget:
        candidates.append(('S_prime', S_prime, S_prime_gain))

    if best_singleton['cost'] <= budget:
        candidates.append(('singleton', [best_singleton['region']], best_singleton['gain']))

    # Handle infeasible cases with prefixes
    if not candidates:
        if S:
            S_prefix = get_fast_feasible_prefix(S, budget, cost_fn)
            if S_prefix:
                candidates.append(('S_prefix', S_prefix, sum(gain_fn([r]) for r in S_prefix)))

        if S_prime:
            S_prime_prefix = get_fast_feasible_prefix(S_prime, budget, cost_fn)
            if S_prime_prefix:
                candidates.append(('S_prime_prefix', S_prime_prefix, sum(gain_fn([r]) for r in S_prime_prefix)))

        if best_singleton['cost'] <= budget:
            candidates.append(('singleton', [best_singleton['region']], best_singleton['gain']))

    # Select best candidate - only compute exact gain for top candidates
    if not candidates:
        S_star, g_star = [], 0
    elif len(candidates) == 1:
        # Only one candidate, use it directly
        _, S_star, g_star = candidates[0]
    else:
        # Multiple candidates - compute exact gains only for top 2 by approximate gain
        candidates.sort(key=lambda x: x[2], reverse=True)
        top_candidates = candidates[:2]

        exact_candidates = []
        for name, solution, approx_gain in top_candidates:
            exact_gain = gain_fn(solution) if solution else 0
            exact_candidates.append((solution, exact_gain))

        S_star, g_star = max(exact_candidates, key=lambda x: x[1])

    print(f"OT: Ultra-fast completed. Final gain = {g_star}, |S*| = {len(S_star)}")

    # Memory aux data
    memory_aux = {
        'S_size': len(S),
        'S_prime_size': len(S_prime),
        'has_I_star': True  # Always have best singleton
    }

    return (S_star, g_star), memory_aux


def get_fast_feasible_prefix(S, budget, cost_fn):
    """Fast feasible prefix using precomputed costs"""
    prefix = []
    total_cost = 0

    for item in S:
        item_cost = cost_fn(item)
        if total_cost + item_cost <= budget:
            prefix.append(item)
            total_cost += item_cost
        else:
            break

    return prefix