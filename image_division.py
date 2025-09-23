import numpy as np


def image_division(images, saliency_maps, N, m):
    """
    Chia ảnh thành N×N patch, chọn m patch quan trọng nhất theo saliency.

    Args:
        images: list ảnh gốc (H,W,C)
        saliency_maps: list saliency map (H,W)
        N: số patch mỗi chiều
        m: số patch được chọn trên mỗi ảnh

    Returns:
        subregions: list dict chứa mask, saliency và image reference
    """
    subregions = []
    for idx, (I, A) in enumerate(zip(images, saliency_maps)):
        # Ensure consistent shapes
        if len(I.shape) == 3 and I.shape[2] == 3:  # (H,W,C)
            h, w, c = I.shape
        else:
            raise ValueError(f"Expected image shape (H,W,C), got {I.shape}")

        if len(A.shape) == 2:  # (H,W)
            h_sal, w_sal = A.shape
        else:
            raise ValueError(f"Expected saliency shape (H,W), got {A.shape}")

        # Use saliency map dimensions for patch division
        ph, pw = h_sal // N, w_sal // N
        patches = []

        for i in range(N):
            for j in range(N):
                y0, y1 = i * ph, (i + 1) * ph
                x0, x1 = j * pw, (j + 1) * pw
                mask = np.zeros((h_sal, w_sal), dtype=np.float32)  # Ensure float32 and 2D
                mask[y0:y1, x0:x1] = 1
                sal_val = (A * mask).sum()

                patches.append({
                    "id": f"{idx}_{i}_{j}",
                    "mask": mask,
                    "saliency": A,
                    "image": I,  # Keep original (H,W,C) format
                    "gain_val": sal_val
                })

        # chọn m patch có saliency cao nhất
        patches.sort(key=lambda x: x["gain_val"], reverse=True)
        subregions.extend(patches[:m])

    return subregions