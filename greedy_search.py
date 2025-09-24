import numpy as np
from image_division import image_division


def marginal_gain(element, current_set, gain_fn):
    """Calculate marginal gain: g(I^M|S) = g(S ∪ {I^M}) - g(S)"""
    if not current_set:
        return gain_fn([element])

    gain_with = gain_fn(current_set + [element])
    gain_without = gain_fn(current_set)
    return gain_with - gain_without


def Greedy_Search(images, saliency_maps, N, m, budget, cost_fn, gain_fn):
    """
    Greedy Search algorithm following Algorithm GS (Algorithm 4) from paper.
    Returns: ((solution_set, gain), memory_aux_data)
    """
    print("Greedy: Starting Algorithm GS")

    # Line 1: V ← ID(I, N, A, m) - Call the ID subroutine
    V = image_division(images, saliency_maps, N, m)
    print(f"Greedy: Generated {len(V)} subregions")

    # Line 2: S ← ∅
    S = []

    # Line 3: U ← V
    U = list(V)  # Copy of V for remaining elements

    iteration = 0

    # Line 4: repeat
    while True:
        iteration += 1
        print(f"Greedy: Iteration {iteration}, |U|={len(U)}, |S|={len(S)}")

        # Line 5: I_t^M ← arg max_{I^M ∈ U} g(I^M|S + I^M)/c(I^M)
        # Choose which I^M has the highest density gain
        best_region = None
        best_density = -1

        for I_M in U:
            # Calculate marginal gain g(I^M|S)
            marginal_g = marginal_gain(I_M, S, gain_fn)
            cost = cost_fn(I_M)

            if cost > 0:  # Avoid division by zero
                density = marginal_g / cost
                if density > best_density:
                    best_density = density
                    best_region = I_M

        # If no valid region found, break
        if best_region is None:
            print("Greedy: No more valid regions found")
            break

        # Line 6: if c(S + I_t^M) ≤ B then
        current_cost = sum(cost_fn(x) for x in S)
        new_region_cost = cost_fn(best_region)

        if current_cost + new_region_cost <= budget:
            # Line 7: S = S + I_t^M
            S.append(best_region)
            print(f"Greedy: Added region {best_region['id']}, density={best_density:.3f}")
        else:
            print("Greedy: Budget constraint violated, stopping")
            break

        # Line 8: U ← U \ {I_t^M}
        U.remove(best_region)

        # Line 9: until U = ∅
        if not U:
            print("Greedy: All regions processed")
            break

    # Line 10: return S
    total_gain = gain_fn(S) if S else 0
    total_cost = sum(cost_fn(x) for x in S)

    print(f"Greedy: Completed. |S|={len(S)}, gain={total_gain:.3f}, cost={total_cost}")

    # Memory aux data for Greedy (no additional structures needed)
    memory_aux = {}

    return (S, total_gain), memory_aux