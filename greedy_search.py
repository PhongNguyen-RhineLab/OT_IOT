import numpy as np


def greedy_selection(images, costs, gains, budget):
    """
    Greedy Search algorithm (GS) for submodular subset selection.

    Args:
        images: list các sub-region (giả sử đã chia ảnh thành các vùng con)
        costs: dict {i: cost} chi phí của mỗi sub-region
        gains: dict {i: gain} lợi ích (giá trị hàm submodular) của mỗi sub-region
        budget: tổng ngân sách B

    Returns:
        S: tập hợp sub-region được chọn
    """
    S = set()
    total_cost = 0

    while total_cost <= budget:
        best_item = None
        best_ratio = -1

        # duyệt qua các vùng chưa chọn
        for i in images:
            if i in S:
                continue
            # kiểm tra nếu thêm i thì có vượt budget không
            if total_cost + costs[i] > budget:
                continue

            # tính tỷ lệ "mật độ lợi ích"
            ratio = gains[i] / costs[i]
            if ratio > best_ratio:
                best_ratio = ratio
                best_item = i

        if best_item is None:
            break  # không chọn thêm được nữa

        # thêm vùng tốt nhất vào tập chọn
        S.add(best_item)
        total_cost += costs[best_item]

    return S


# ------------------ Ví dụ giả lập ------------------ #
# giả sử có 5 sub-region: I1..I5
images = ["I1", "I2", "I3", "I4", "I5"]

# chi phí chọn mỗi vùng
costs = {"I1": 3, "I2": 2, "I3": 4, "I4": 1, "I5": 2}

# lợi ích (gain) mỗi vùng (ví dụ tính từ saliency map)
gains = {"I1": 9, "I2": 5, "I3": 6, "I4": 2, "I5": 4}

# ngân sách
budget = 6

selected = greedy_selection(images, costs, gains, budget)
print("Các sub-region được chọn:", selected)
