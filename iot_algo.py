import math
from ot_algo import OT_algorithm
from image_division import image_division


def marginal_gain(element, current_set, gain_fn):
    """Calculate marginal gain: g(element|current_set)"""
    gain_with = sum(gain_fn(x) for x in current_set + [element])
    gain_without = sum(gain_fn(x) for x in current_set)
    return gain_with - gain_without


def IOT_algorithm(images, saliency_maps, N, m, budget, eps, cost_fn, gain_fn):
    """
    Improved Online Threshold (IOT) algorithm following Algorithm 3 in paper.
    """
    # Line 3: First pass with OT
    Sb, M = OT_algorithm(images, saliency_maps, N, m, budget, cost_fn, gain_fn)
    M = sum(gain_fn(x) for x in Sb)  # Get the gain value

    # Line 2: Set epsilon prime
    eps_prime = eps / 5

    # Line 4: Generate threshold set T
    T = []
    i = 0
    lower_bound = (1 - eps_prime) * eps_prime * M / (2 * budget)
    upper_bound = 4 * M / (eps_prime * budget)

    tau = lower_bound
    while tau <= upper_bound:
        T.append(tau)
        tau *= (1 + eps_prime)  # Geometric progression

    # Initialize candidate sets for each threshold
    candidates = {}
    for tau in T:
        candidates[tau] = {'S_tau': [], 'S_prime_tau': []}

    # Line 5-7: Second pass through data stream
    V = image_division(images, saliency_maps, N, m)

    for I_M in V:
        # Line 8-12: For each threshold
        for tau in T:
            S_tau = candidates[tau]['S_tau']
            S_prime_tau = candidates[tau]['S_prime_tau']

            # Line 10: Choose candidate with higher marginal gain
            gain_S_tau = marginal_gain(I_M, S_tau, gain_fn) if S_tau else 0
            gain_S_prime_tau = marginal_gain(I_M, S_prime_tau, gain_fn) if S_prime_tau else 0

            if gain_S_tau >= gain_S_prime_tau:
                X_tau = S_tau
                target_key = 'S_tau'
            else:
                X_tau = S_prime_tau
                target_key = 'S_prime_tau'

            # Line 11: Threshold condition and budget constraint
            marginal_g = marginal_gain(I_M, X_tau, gain_fn)
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
        return [], 0

    S_star = max(all_candidates, key=lambda X: sum(gain_fn(x) for x in X))
    return S_star, sum(gain_fn(x) for x in S_star)