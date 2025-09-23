import numpy as np
from image_division import image_division


def Greedy_Search(images, saliency_maps, N, m, budget, cost_fn, gain_fn):
    """
    Greedy Selection algorithm with memory tracking.
    Returns: ((solution_set, gain), memory_aux_data)
    """
    V = image_division(images, saliency_maps, N, m)
    S = []
    total_cost = 0

    while True:
        candidates = []
        for r in V:
            if r not in S:
                c, g = cost_fn(r), gain_fn(r)
                if total_cost + c <= budget:
                    candidates.append((g / c, r))

        if not candidates:
            break

        _, best_region = max(candidates, key=lambda x: x[0])
        S.append(best_region)
        total_cost += cost_fn(best_region)

    total_gain = sum(gain_fn(x) for x in S)

    # Memory aux data for Greedy (no additional structures needed)
    memory_aux = {}

    return (S, total_gain), memory_aux