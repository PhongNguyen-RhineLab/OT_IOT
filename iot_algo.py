import numpy as np
from ot_algo import OT_algorithm
from image_division import image_division

def IOT_algorithm(images, saliency_maps, N, m, budget, eps, cost_fn, gain_fn):
    """
    Improved Online Threshold (IOT) algorithm (Algorithm 3).
    Trả về (S*, g(S*)).
    """
    S_star = []
    eps_prime = eps / 5

    # Gọi OT để lấy Sb và g(Sb)
    Sb, g_Sb = OT_algorithm(images, saliency_maps, N, m, budget, cost_fn, gain_fn)

    # Sinh tập ngưỡng T theo paper
    T = []
    i = 0
    while True:
        tau = (1 - eps_prime) ** i * (g_Sb * eps_prime) / (2 * budget)
        if tau >= (1 - eps_prime) * g_Sb / (2 * budget) and tau <= (4 * g_Sb) / (eps_prime * budget):
            T.append(tau)
            i += 1
        else:
            break

    # mỗi τ có một tập X_tau
    X_dict = {tau: [] for tau in T}

    # Stream 2: xử lý ảnh
    for I, A in zip(images, saliency_maps):
        V = image_division([I], [A], N, m)

        for I_M in V:  # mỗi sub-region
            for tau in T:
                X_tau = X_dict[tau]

                # kiểm tra điều kiện thêm I_M
                current_cost = sum(cost_fn(x) for x in X_tau)
                if (gain_fn(I_M) / cost_fn(I_M)) >= tau and current_cost + cost_fn(I_M) <= budget:
                    X_dict[tau] = X_tau + [I_M]

                # cập nhật S_star
                if sum(gain_fn(x) for x in X_dict[tau]) > sum(gain_fn(x) for x in S_star):
                    S_star = X_dict[tau]

    # cuối cùng chọn tốt nhất giữa Sb và S_star
    final_set = max([Sb, S_star], key=lambda X: sum(gain_fn(x) for x in X))
    return final_set, sum(gain_fn(x) for x in final_set)
