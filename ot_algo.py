from image_division import image_division

def OT_algorithm(images, saliency_maps, N, m, budget, cost_fn, gain_fn):
    """
    Online Threshold (OT) algorithm.
    """
    V = image_division(images, saliency_maps, N, m)
    S, S_prime, I_star = [], [], None

    for I_M in V:
        candidates = [S, S_prime]
        g_values = [sum(gain_fn(x) for x in S), sum(gain_fn(x) for x in S_prime)]
        Sd = candidates[g_values.index(max(g_values))]

        if gain_fn(I_M) / cost_fn(I_M) >= sum(gain_fn(x) for x in Sd) / max(budget, 1):
            Sd = Sd + [I_M]

        all_candidates = S + S_prime + ([I_star] if I_star else [])
        if all_candidates:
            I_star = max(all_candidates, key=lambda x: gain_fn(x))

    # Chọn tập cuối
    candidates = [S, S_prime, [I_star] if I_star else []]
    feasible = [X for X in candidates if sum(cost_fn(x) for x in X) <= budget]
    if not feasible:
        return [], 0
    S_star = max(feasible, key=lambda X: sum(gain_fn(x) for x in X))

    return S_star, sum(gain_fn(x) for x in S_star)
