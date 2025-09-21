import numpy as np
from ot_algo import OT_algorithm
from image_division import image_division

def IOT_algorithm(images, saliency_maps, N, m, budget, eps, cost_fn, gain_fn):
    """
    Improved Online Threshold (IOT) algorithm.

    Args:
        images: list các ảnh gốc
        saliency_maps: list saliency maps
        N: số patch chia theo mỗi chiều
        m: số sub-region mỗi ảnh
        budget: ngân sách B
        eps: tham số epsilon (0 < eps < 1)
        cost_fn: hàm chi phí c(I^M)
        gain_fn: hàm lợi ích g(I^M)

    Returns:
        S_star: tập sub-region chọn
    """
    # Khởi tạo
    V, S_T, S_prime, S_star = [], [], [], []
    eps_prime = eps / 5

    # Gọi OT để lấy giá trị tham chiếu (Sb, M)
    Sb, M = OT_algorithm(images, saliency_maps, N, m, budget, cost_fn, gain_fn)

    # Tập ngưỡng τ
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

    # Stream 2: xử lý ảnh mới đến
    for I, A in zip(images, saliency_maps):
        # chia ảnh thành sub-regions bằng ID
        V = image_division([I], [A], N, m)

        for I_M in V:  # mỗi sub-region chưa xét
            for tau in T:
                # tìm tập con tốt nhất hiện tại
                X_tau = max([S_T, S_prime], key=lambda X: sum(gain_fn(x) for x in X))

                # kiểm tra điều kiện thêm I_M
                if (gain_fn(I_M) / cost_fn(I_M)) >= tau and (sum(cost_fn(x) for x in X_tau) + cost_fn(I_M)) <= budget:
                    X_tau = X_tau + [I_M]

                # cập nhật S*
                S_star = max([S_star, X_tau], key=lambda X: sum(gain_fn(x) for x in X))

    # cuối cùng chọn tập tốt nhất giữa Sb và S*
    final_set = max([Sb, S_star], key=lambda X: sum(gain_fn(x) for x in X))
    return final_set
