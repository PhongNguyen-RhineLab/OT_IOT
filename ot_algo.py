from image_division import image_division


def fast_marginal_gain(element, current_set, gain_fn, cached_set_gain=None):
    """Fast marginal gain with caching"""
    if not current_set:
        return gain_fn([element])

    gain_with = gain_fn(current_set + [element])

    # Use cached value if available
    if cached_set_gain is not None:
        gain_without = cached_set_gain
    else:
        gain_without = gain_fn(current_set)

    return gain_with - gain_without


def OT_algorithm(images, saliency_maps, N, m, budget, cost_fn, gain_fn):
    """
    Fast Online Threshold (OT) algorithm with aggressive optimizations.
    Returns: ((solution_set, gain), memory_aux_data)
    """
    print("OT: Starting fast algorithm")
    V = image_division(images, saliency_maps, N, m)
    S, S_prime = [], []
    I_star = None
    I_star_gain = 0

    # Cache current gains to avoid recomputation
    S_gain_cache = 0  # g(S)
    S_prime_gain_cache = 0  # g(S')

    # Precompute singleton gains to avoid repeated calls
    singleton_gains = {}
    for region in V:
        singleton_gains[region['id']] = gain_fn([region])

    processed = 0
    total_regions = len(V)

    for I_M in V:
        processed += 1
        if processed % max(1, total_regions // 5) == 0:
            print(f"OT: Processed {processed}/{total_regions} regions")

        region_cost = cost_fn(I_M)
        singleton_gain = singleton_gains[I_M['id']]

        # Early skip if cost too high
        if region_cost > budget:
            continue

        # Update best singleton (without expensive calls)
        if I_star is None or singleton_gain > I_star_gain:
            I_star = I_M
            I_star_gain = singleton_gain

        # Fast marginal gain calculation using cached values
        marginal_gain_S = fast_marginal_gain(I_M, S, gain_fn, S_gain_cache)
        marginal_gain_S_prime = fast_marginal_gain(I_M, S_prime, gain_fn, S_prime_gain_cache)

        # Choose better candidate
        if marginal_gain_S >= marginal_gain_S_prime:
            S_d = S
            S_d_gain = S_gain_cache
            marginal_g = marginal_gain_S
            is_S = True
        else:
            S_d = S_prime
            S_d_gain = S_prime_gain_cache
            marginal_g = marginal_gain_S_prime
            is_S = False

        # Threshold condition: g(I_M|S_d)/c(I_M) >= g(S_d)/B
        if region_cost > 0 and marginal_g / region_cost >= S_d_gain / budget:
            # Add to selected candidate and update cache
            if is_S:
                S = S + [I_M]
                S_gain_cache += marginal_g  # Incremental update
            else:
                S_prime = S_prime + [I_M]
                S_prime_gain_cache += marginal_g  # Incremental update

    print("OT: Selecting final solution...")

    # Final selection (compute accurate gains for final decision)
    cost_S = sum(cost_fn(x) for x in S) if S else 0
    cost_S_prime = sum(cost_fn(x) for x in S_prime) if S_prime else 0
    cost_I_star = cost_fn(I_star) if I_star else float('inf')

    # Use cached gains for feasible solutions, compute fresh for final accuracy
    candidates = []

    if cost_S <= budget:
        gain_S = gain_fn(S) if S else 0  # Fresh computation for accuracy
        candidates.append((S, gain_S))

    if cost_S_prime <= budget:
        gain_S_prime = gain_fn(S_prime) if S_prime else 0
        candidates.append((S_prime, gain_S_prime))

    if cost_I_star <= budget and I_star:
        candidates.append(([I_star], I_star_gain))

    # Handle infeasible cases
    if not candidates:
        # Both infeasible, use prefixes
        if S:
            S1 = get_feasible_prefix(S, budget, cost_fn)
            if S1:
                candidates.append((S1, gain_fn(S1)))

        if S_prime:
            S2 = get_feasible_prefix(S_prime, budget, cost_fn)
            if S2:
                candidates.append((S2, gain_fn(S2)))

        if cost_I_star <= budget and I_star:
            candidates.append(([I_star], I_star_gain))

    # Select best candidate
    if candidates:
        S_star, g_star = max(candidates, key=lambda x: x[1])
    else:
        S_star, g_star = [], 0

    print(f"OT: Completed. Final gain = {g_star}, |S*| = {len(S_star)}")

    # Memory aux data for OT
    memory_aux = {
        'S_size': len(S),
        'S_prime_size': len(S_prime),
        'has_I_star': I_star is not None
    }

    return (S_star, g_star), memory_aux


def get_feasible_prefix(S, budget, cost_fn):
    """Get largest prefix of S that satisfies budget constraint"""
    prefix = []
    total_cost = 0
    for item in S:
        if total_cost + cost_fn(item) <= budget:
            prefix.append(item)
            total_cost += cost_fn(item)
        else:
            break
    return prefix