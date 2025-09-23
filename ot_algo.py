from image_division import image_division


def marginal_gain(element, current_set, gain_fn):
    """Calculate marginal gain efficiently: g(element|current_set)"""
    if not current_set:
        return gain_fn([element])

    gain_with = gain_fn(current_set + [element])
    gain_without = gain_fn(current_set)
    return gain_with - gain_without


def OT_algorithm(images, saliency_maps, N, m, budget, cost_fn, gain_fn):
    """
    Optimized Online Threshold (OT) algorithm.
    Follows Algorithm 2 but with optimizations for speed.
    """
    print("OT: Starting algorithm")
    V = image_division(images, saliency_maps, N, m)
    S, S_prime = [], []
    I_star = None
    I_star_gain = 0

    # Cache for gain computations to avoid recomputation
    S_gain_cache = 0  # g(S)
    S_prime_gain_cache = 0  # g(S')

    processed = 0
    for I_M in V:
        processed += 1
        if processed % max(1, len(V) // 10) == 0:
            print(f"OT: Processed {processed}/{len(V)} regions")

        # Line 6: Choose candidate with higher marginal gain
        # Calculate marginal gains efficiently
        marginal_gain_S = marginal_gain(I_M, S, gain_fn)
        marginal_gain_S_prime = marginal_gain(I_M, S_prime, gain_fn)

        # Choose the better candidate
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

        # Line 7: Threshold condition g(I_M|S_d)/c(I_M) >= g(S_d)/B
        region_cost = cost_fn(I_M)
        if region_cost > 0 and marginal_g / region_cost >= S_d_gain / budget:
            # Line 8: Add to selected candidate
            if is_S:
                S = S + [I_M]
                S_gain_cache += marginal_g  # Update cache efficiently
            else:
                S_prime = S_prime + [I_M]
                S_prime_gain_cache += marginal_g  # Update cache efficiently

        # Line 9: Update best singleton
        singleton_gain = gain_fn([I_M])
        if I_star is None or singleton_gain > I_star_gain:
            I_star = I_M
            I_star_gain = singleton_gain

    print("OT: Selecting final solution...")

    # Lines 10-18: Final selection based on budget constraints
    cost_S = sum(cost_fn(x) for x in S)
    cost_S_prime = sum(cost_fn(x) for x in S_prime)
    cost_I_star = cost_fn(I_star) if I_star else float('inf')

    # Get actual gains (not cached, for final accuracy)
    gain_S = gain_fn(S) if S else 0
    gain_S_prime = gain_fn(S_prime) if S_prime else 0
    gain_I_star = I_star_gain if I_star else 0

    if cost_S <= budget and cost_S_prime <= budget:
        # Line 11: Both feasible
        candidates = [(S, gain_S), (S_prime, gain_S_prime)]
        if cost_I_star <= budget:
            candidates.append(([I_star], gain_I_star))

        S_star, g_star = max(candidates, key=lambda x: x[1])

    elif cost_S > budget and cost_S_prime > budget:
        # Lines 13-15: Both infeasible, take prefixes
        S1 = get_feasible_prefix(S, budget, cost_fn)
        S2 = get_feasible_prefix(S_prime, budget, cost_fn)

        candidates = [(S1, gain_fn(S1) if S1 else 0),
                      (S2, gain_fn(S2) if S2 else 0)]
        if cost_I_star <= budget:
            candidates.append(([I_star], gain_I_star))

        S_star, g_star = max(candidates, key=lambda x: x[1])

    else:
        # Lines 16-18: One feasible, one infeasible
        if cost_S <= budget:
            feasible, feasible_gain = S, gain_S
            infeasible = S_prime
        else:
            feasible, feasible_gain = S_prime, gain_S_prime
            infeasible = S

        infeasible_prefix = get_feasible_prefix(infeasible, budget, cost_fn)

        candidates = [(feasible, feasible_gain),
                      (infeasible_prefix, gain_fn(infeasible_prefix) if infeasible_prefix else 0)]
        if cost_I_star <= budget:
            candidates.append(([I_star], gain_I_star))

        S_star, g_star = max(candidates, key=lambda x: x[1])

    print(f"OT: Completed. Final gain = {g_star}, |S*| = {len(S_star)}")
    return S_star, g_star


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