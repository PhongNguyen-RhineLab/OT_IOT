"""
image_division.py - Algorithm ID Implementation with Proper Cost Scaling
Paper: "Online approximate algorithms for Object detection under Budget allocation"
"""

import numpy as np


def image_division(images, saliency_maps, N, m):
    """
    Algorithm ID from paper with proper cost scaling for budget constraints.

    Args:
        images: list ảnh gốc (H,W,C)
        saliency_maps: list saliency map (H,W)
        N: số patch mỗi chiều (N×N grid)
        m: số subregion được chọn trên mỗi ảnh

    Returns:
        subregions: list dict chứa subregion information
    """
    print(f"Image Division (Algorithm ID): {len(images)} images, N={N}, m={m}")

    subregions = []
    total_patches_created = 0

    for idx, (I, A) in enumerate(zip(images, saliency_maps)):
        # Validate input shapes
        if len(I.shape) == 3 and I.shape[2] == 3:  # (H,W,C)
            h, w, c = I.shape
        else:
            raise ValueError(f"Expected image shape (H,W,C), got {I.shape}")

        if len(A.shape) == 2:  # (H,W)
            h_sal, w_sal = A.shape
        else:
            raise ValueError(f"Expected saliency shape (H,W), got {A.shape}")

        # Debug info for first image
        if idx == 0:
            print(f"  Sample image: {I.shape}, saliency: {A.shape}")
            print(f"  Saliency range: [{A.min():.3f}, {A.max():.3f}]")

        # Calculate patch dimensions
        ph, pw = h_sal // N, w_sal // N
        d = max(1, (N * N) // m)  # patches per subregion (approximately)

        if idx == 0:
            print(f"  Patch size: {ph}×{pw} = {ph * pw} pixels")
            print(f"  Target patches per subregion: {d}")

        # Create all N×N patches first
        patches = []
        for i in range(N):
            for j in range(N):
                y0, y1 = i * ph, (i + 1) * ph
                x0, x1 = j * pw, (j + 1) * pw

                # Create binary mask for this patch
                mask = np.zeros((h_sal, w_sal), dtype=np.float32)
                mask[y0:y1, x0:x1] = 1.0

                # Calculate saliency importance for ranking
                sal_val = (A * mask).sum()

                patches.append({
                    'patch_idx': i * N + j,
                    'position': (i, j),
                    'mask': mask,
                    'saliency_val': sal_val,
                    'bounds': (y0, y1, x0, x1)
                })

        # Sort patches by saliency importance (descending)
        patches.sort(key=lambda x: x['saliency_val'], reverse=True)

        # Create m subregions following Algorithm ID logic
        for l in range(1, m + 1):  # Line 5: l = 1 to m
            # Select patches for this subregion based on ranking
            start_idx = (l - 1) * d
            end_idx = min(l * d, len(patches))

            if start_idx >= len(patches):
                # Not enough patches, create single patch subregion
                selected_patches = [patches[l - 1]] if l - 1 < len(patches) else [patches[-1]]
            else:
                selected_patches = patches[start_idx:end_idx]

            # Create subregion mask by combining selected patches
            I_M = np.zeros((h_sal, w_sal), dtype=np.float32)
            for patch in selected_patches:
                I_M += patch['mask']

            # Clip to avoid overlap issues
            I_M = np.clip(I_M, 0, 1)

            # CRITICAL: Normalize cost to reasonable range
            # Cost = number of patches in this subregion (instead of pixel count)
            actual_patches = len(selected_patches)
            cost_normalized_mask = I_M * (actual_patches / max(I_M.sum(), 1e-6))

            # Calculate gain using original mask  
            gain_value = (A * I_M).sum()

            subregion = {
                "id": f"{idx}_{l}",
                "mask": cost_normalized_mask,  # For cost calculation
                "original_mask": I_M,  # For gain calculation
                "saliency": A,
                "image": I,
                "gain_val": float(gain_value),
                "patch_count": actual_patches,
                "selected_patches": len(selected_patches)
            }

            subregions.append(subregion)
            total_patches_created += actual_patches

    if subregions:
        sample_cost = subregions[0]["mask"].sum()
        sample_gain = subregions[0]["gain_val"]
        print(f"  Sample subregion: cost={sample_cost:.3f}, gain={sample_gain:.1f}")

    print(f"Image Division completed: {len(subregions)} subregions")
    print(f"Average patches per subregion: {total_patches_created / len(subregions):.1f}")

    return subregions


def test_image_division():
    """Test function to verify image division works correctly"""
    print("=== TESTING IMAGE DIVISION ===")

    # Create dummy test data
    test_images = [np.random.rand(224, 224, 3) for _ in range(2)]
    test_saliency = [np.random.rand(224, 224) for _ in range(2)]

    # Test with different parameters
    test_configs = [
        (4, 5),  # 4×4 grid, 5 subregions per image
        (4, 8),  # 4×4 grid, 8 subregions per image
        (6, 10),  # 6×6 grid, 10 subregions per image
    ]

    for N, m in test_configs:
        print(f"\nTesting N={N}, m={m}:")
        subregions = image_division(test_images, test_saliency, N, m)

        # Analyze results
        costs = [r["mask"].sum() for r in subregions]
        gains = [r["gain_val"] for r in subregions]

        print(f"  Generated {len(subregions)} subregions")
        print(f"  Cost range: [{min(costs):.3f}, {max(costs):.3f}]")
        print(f"  Gain range: [{min(gains):.1f}, {max(gains):.1f}]")

        # Check feasibility with different budgets
        budgets = [5, 10, 20, 50]
        for budget in budgets:
            feasible = sum(1 for cost in costs if cost <= budget)
            print(f"    Budget {budget:2d}: {feasible:2d}/{len(costs)} feasible ({feasible / len(costs) * 100:.0f}%)")


if __name__ == "__main__":
    test_image_division()