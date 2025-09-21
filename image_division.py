import numpy as np
from skimage.transform import resize


def image_division(images, saliency_maps, N, m):
    """
    Image Division algorithm (ID)

    Args:
        images: list các ảnh gốc (mỗi ảnh có shape (h, w, 3))
        saliency_maps: list saliency map tương ứng (shape (h, w))
        N: số patch chia theo mỗi chiều (ảnh chia thành N x N ô)
        m: số sub-region cần tạo

    Returns:
        V: danh sách các sub-region (dưới dạng ảnh mask)
    """
    V = []

    for I, A in zip(images, saliency_maps):
        h, w, _ = I.shape

        # resize saliency map về N x N
        A_resized = resize(A, (N, N), preserve_range=True)

        # số patch mỗi sub-region
        d = (N * N) // m

        # rank (flatten rồi sort theo giá trị saliency giảm dần)
        flat_idx = np.argsort(A_resized, axis=None)[::-1]  # chỉ số đã sắp xếp giảm dần

        # Tạo m sub-region
        for l in range(m):
            I_M = np.zeros_like(I)

            for j in range(d * l, d * (l + 1)):
                idx = flat_idx[j]
                # chuyển chỉ số 1D -> 2D
                i_r, i_c = np.unravel_index(idx, (N, N))

                # xác định vùng ảnh gốc tương ứng với patch này
                row_start = int(i_r * h / N)
                row_end = int((i_r + 1) * h / N)
                col_start = int(i_c * w / N)
                col_end = int((i_c + 1) * w / N)

                # copy patch từ ảnh gốc vào sub-region
                I_M[row_start:row_end, col_start:col_end, :] = I[row_start:row_end, col_start:col_end, :]

            V.append(I_M)

    return V


# ------------------ Ví dụ giả lập ------------------ #
if __name__ == "__main__":
    # tạo ảnh giả (100x100, 3 kênh)
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    # saliency map giả (100x100)
    saliency = np.random.rand(100, 100)

    # chạy ID algorithm
    subregions = image_division([img], [saliency], N=10, m=5)
    print("Số sub-region tạo ra:", len(subregions))
