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
        h, w = A.shape
        ph, pw = h // N, w // N
        patches = []

        for i in range(N):
            for j in range(N):
                y0, y1 = i * ph, (i + 1) * ph
                x0, x1 = j * pw, (j + 1) * pw
                mask = np.zeros_like(A, dtype=np.uint8)
                mask[y0:y1, x0:x1] = 1
                sal_val = (A * mask).sum()

                patches.append({
                    "id": f"{idx}_{i}_{j}",
                    "mask": mask,
                    "saliency": A,
                    "image": I,  # ADD: Original image reference for submodular function
                    "gain_val": sal_val
                })

        # chọn m patch có saliency cao nhất
        patches.sort(key=lambda x: x["gain_val"], reverse=True)
        subregions.extend(patches[:m])

    return subregions