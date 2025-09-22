from image_division import image_division


def marginal_gain(element, current_set, gain_fn):
    """Calculate marginal gain: g(element|current_set)"""
    gain_with = sum(gain_fn(x) for x in current_set + [element])
    gain_without = sum(gain_fn(x) for x in current_set)
    return gain_with - gain_without


def OT_algorithm(images, saliency_maps, N, m, budget, cost_fn, gain_fn):
    """
    Online Threshold (OT) algorithm following Algorithm 2 in paper.
    """
    V = image_division(images, saliency_maps, N, m)
    S, S_prime = [], []
    I_star = None

    for I_M in V:
        # Line 6: Choose candidate with higher marginal gain
        if not S and not S_prime:
            S_d = S  # Default to S if both empty
        else:
            gain_S = marginal_gain(I_M, S, gain_fn) if S else 0
            gain_S_prime = marginal_gain(I_M, S_prime, gain_fn) if S_prime else 0
            S_d = S if gain_S >= gain_S_prime else S_prime

        # Line 7: Threshold condition
        marginal_g = marginal_gain(I_M, S_d, gain_fn)
        current_gain = sum(gain_fn(x) for x in S_d)

        if marginal_g / cost_fn(I_M) >= current_gain / budget:
            # Line 8: Add to selected candidate
            if S_d is S:
                S = S + [I_M]
            else:
                S_prime = S_prime + [I_M]

        # Line 9: Update best singleton
        if I_star is None or gain_fn(I_M) > gain_fn(I_star):
            I_star = I_M

    # Lines 10-18: Final selection based on budget constraints
    cost_S = sum(cost_fn(x) for x in S)
    cost_S_prime = sum(cost_fn(x) for x in S_prime)

    if cost_S <= budget and cost_S_prime <= budget:
        # Line 11: Both feasible
        candidates = [S, S_prime, [I_star] if I_star else []]
        S_star = max(candidates, key=lambda X: sum(gain_fn(x) for x in X))
    elif cost_S > budget and cost_S_prime > budget:
        # Lines 13-15: Both infeasible, take prefixes
        S1 = get_feasible_prefix(S, budget, cost_fn)
        S2 = get_feasible_prefix(S_prime, budget, cost_fn)
        candidates = [S1, S2, [I_star] if I_star else []]
        S_star = max(candidates, key=lambda X: sum(gain_fn(x) for x in X))
    else:
        # Lines 16-18: One feasible, one infeasible
        if cost_S <= budget:
            feasible, infeasible = S, S_prime
        else:
            feasible, infeasible = S_prime, S

        infeasible_prefix = get_feasible_prefix(infeasible, budget, cost_fn)
        candidates = [feasible, infeasible_prefix, [I_star] if I_star else []]
        S_star = max(candidates, key=lambda X: sum(gain_fn(x) for x in X))

    return S_star, sum(gain_fn(x) for x in S_star)


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