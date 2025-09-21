from ot_algo import OT_algorithm
from image_division import image_division

def IOT_algorithm(images, saliency_maps, N, m, budget, eps, cost_fn, gain_fn):
    """
    Improved Online Threshold (IOT) algorithm.
    """
    Sb, M = OT_algorithm(images, saliency_maps, N, m, budget, cost_fn, gain_fn)
    eps_prime = eps / 5
    T = []
    i = 0
    while True:
        tau = (1 - eps_prime) ** i
        tau *= (M * eps_prime) / (2 * budget)
        if tau < (M * 4) / (eps_prime * budget):
            T.append(tau)
            i += 1
        else:
            break

    S_star = []
    for I_M in image_division(images, saliency_maps, N, m):
        for tau in T:
            X_tau = max([S_star, Sb], key=lambda X: sum(gain_fn(x) for x in X))
            if (gain_fn(I_M) / cost_fn(I_M)) >= tau and \
               (sum(cost_fn(x) for x in X_tau) + cost_fn(I_M)) <= budget:
                X_tau = X_tau + [I_M]
            S_star = max([S_star, X_tau], key=lambda X: sum(gain_fn(x) for x in X))

    final_set = max([Sb, S_star], key=lambda X: sum(gain_fn(x) for x in X))
    return final_set, sum(gain_fn(x) for x in final_set)
