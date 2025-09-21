import numpy as np
from image_division import image_division

def OT_algorithm(images, saliency_maps, N, m, budget, cost_fn, gain_fn):
    """
    Online Threshold (OT) algorithm for interpretable region discovery (Algorithm 2).
    Trả về (S*, g(S*)).
    """
    V, S, S_prime = [], [], []
    I_star = None
    S_star = []

    for I, A in zip(images, saliency_maps):
        # chia ảnh thành các sub-region theo ID
        V = image_division([I], [A], N, m)

        for I_M in V:
            # chọn Sd = S hoặc S_prime có gain lớn hơn
            if sum(gain_fn(x) for x in S) >= sum(gain_fn(x) for x in S_prime):
                Sd = S
            else:
                Sd = S_prime

            # marginal gain khi thêm I_M
            g_before = sum(gain_fn(x) for x in Sd)
            marginal_gain = gain_fn(I_M)

            if marginal_gain / cost_fn(I_M) >= g_before / budget:
                Sd.append(I_M)

            # cập nhật I_star (subregion đơn lẻ có gain lớn nhất)
            if I_star is None or gain_fn(I_M) > gain_fn(I_star):
                I_star = I_M

        # Kiểm tra ràng buộc ngân sách
        cost_S = sum(cost_fn(x) for x in S)
        cost_Sp = sum(cost_fn(x) for x in S_prime)

        if cost_S <= budget and cost_Sp <= budget:
            candidates = [S, S_prime, [I_star] if I_star else []]

        elif cost_S >= budget and cost_Sp >= budget:
            # lọc lại để không vượt budget
            def filter_budget(lst):
                acc, result = 0, []
                for x in lst:
                    if acc + cost_fn(x) <= budget:
                        result.append(x)
                        acc += cost_fn(x)
                return result
            S1 = filter_budget(S)
            S2 = filter_budget(S_prime)
            candidates = [S1, S2]

        else:  # một trong hai vượt budget
            def filter_budget(lst):
                acc, result = 0, []
                for x in lst:
                    if acc + cost_fn(x) <= budget:
                        result.append(x)
                        acc += cost_fn(x)
                return result
            S1 = filter_budget(S)
            S2 = filter_budget(S_prime)
            candidates = [S1, S2, [I_star] if I_star else []]

        # chọn tập tốt nhất
        S_star = max(candidates, key=lambda X: sum(gain_fn(x) for x in X))

    return S_star, sum(gain_fn(x) for x in S_star)
