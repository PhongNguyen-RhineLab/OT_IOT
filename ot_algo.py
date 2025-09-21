import numpy as np
from image_division import image_division

def OT_algorithm(images, saliency_maps, N, m, budget, cost_fn, gain_fn):
    """
    Online Threshold (OT) algorithm for interpretable region discovery.

    Args:
        images: list các ảnh gốc
        saliency_maps: list saliency maps tương ứng
        N: số patch chia theo mỗi chiều
        m: số sub-region cho mỗi ảnh
        budget: ngân sách B
        cost_fn: hàm tính chi phí c(I^M)
        gain_fn: hàm tính lợi ích g(I^M)

    Returns:
        (S_star, g(S_star)) tập sub-region được chọn và tổng gain
    """

    # tập ban đầu rỗng
    V, S, S_prime, I_star = [], [], [], None

    for I, A in zip(images, saliency_maps):
        # chia ảnh thành các sub-region theo ID
        V += image_division([I], [A], N, m)  # dùng hàm ID đã viết trước đó

        # duyệt qua từng sub-region
        for I_M in V:
            # tìm tập con Sd tốt nhất hiện tại
            candidates = [S, S_prime]
            g_values = [sum(gain_fn(x) for x in S), sum(gain_fn(x) for x in S_prime)]
            Sd = candidates[np.argmax(g_values)]

            # kiểm tra tỷ lệ gain/cost
            if gain_fn(I_M) / cost_fn(I_M) >= sum(gain_fn(x) for x in Sd) / budget:
                Sd = Sd + [I_M]

            # cập nhật I_star
            all_candidates = (S + S_prime + ([I_star] if I_star else []))
            if all_candidates:
                I_star = max(all_candidates, key=lambda x: gain_fn(x))

        # kiểm tra ràng buộc ngân sách
        cost_S = sum(cost_fn(x) for x in S)
        cost_Sp = sum(cost_fn(x) for x in S_prime)

        if cost_S <= budget and cost_Sp <= budget:
            # chọn tập có gain lớn nhất
            candidates = [S, S_prime, [I_star] if I_star else []]
            S_star = max(candidates, key=lambda X: sum(gain_fn(x) for x in X))

        elif cost_S >= budget and cost_Sp >= budget:
            # chọn tập con "gần ngân sách nhất" nhưng không vượt quá
            S1 = [x for x in S if sum(cost_fn(y) for y in (S1+[x])) <= budget]
            S2 = [x for x in S_prime if sum(cost_fn(y) for y in (S2+[x])) <= budget]
            candidates = [S1, S2]
            S_star = max(candidates, key=lambda X: sum(gain_fn(x) for x in X))

        else:
            # một trong hai vượt ngân sách, thử hoán đổi
            S1 = [x for x in S if sum(cost_fn(y) for y in (S1+[x])) <= budget]
            S2 = [x for x in S_prime if sum(cost_fn(y) for y in (S2+[x])) <= budget]
            candidates = [S1, S2, [I_star] if I_star else []]
            S_star = max(candidates, key=lambda X: sum(gain_fn(x) for x in X))

    return S_star, sum(gain_fn(x) for x in S_star)
