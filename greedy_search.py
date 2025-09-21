import numpy as np
from image_division import image_division

def Greedy_Search(images, saliency_maps, N, m, budget, cost_fn, gain_fn):
    """
    Greedy Search (GS) algorithm for interpretable region discovery.
    Trả về (S, g(S)).
    """
    # chia toàn bộ ảnh thành sub-region
    V = []
    for I, A in zip(images, saliency_maps):
        V += image_division([I], [A], N, m)

    S = []
    total_cost = 0

    while True:
        # tìm region có marginal gain/cost cao nhất
        candidates = []
        for I_M in V:
            if I_M not in S:
                c = cost_fn(I_M)
                g = gain_fn(I_M)
                if total_cost + c <= budget:  # không vượt budget
                    candidates.append((g / c, I_M))

        if not candidates:  # không chọn thêm được nữa
            break

        # chọn I_M tốt nhất
        _, best_region = max(candidates, key=lambda x: x[0])
        S.append(best_region)
        total_cost += cost_fn(best_region)

    return S, sum(gain_fn(x) for x in S)
