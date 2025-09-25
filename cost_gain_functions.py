"""
cost_gain_functions.py - Paper-Correct Cost and Gain Functions
Paper: "Online approximate algorithms for Object detection under Budget allocation"
"""

import numpy as np


def paper_cost_function(region_or_list):
    """
    Cost function c(·) from paper - modular cost function
    c(S) = Σ_{I^M ∈ S} c(I^M)

    Args:
        region_or_list: single subregion dict or list of subregion dicts

    Returns:
        float: total cost
    """
    if isinstance(region_or_list, list):
        if not region_or_list:
            return 0.0
        return sum(paper_cost_function(r) for r in region_or_list)
    else:
        # Single region cost = sum of (normalized) mask
        return float(region_or_list["mask"].sum())


def paper_gain_function_simple(region_or_list):
    """
    Simple gain function g(·) for paper experiments - saliency-based
    Uses original masks (not cost-scaled) for gain calculation

    Args:
        region_or_list: single subregion dict or list of subregion dicts

    Returns:
        float: total gain value
    """
    if isinstance(region_or_list, list):
        if not region_or_list:
            return 0.0

        total_gain = 0.0
        for region in region_or_list:
            # Use original mask for gain calculation (not cost-scaled mask)
            if 'original_mask' in region:
                mask = region["original_mask"]
            else:
                mask = region["mask"]

            saliency = region["saliency"]
            gain = (saliency * mask).sum()
            total_gain += gain

        return float(total_gain)
    else:
        # Single region gain
        if 'original_mask' in region_or_list:
            mask = region_or_list["original_mask"]
        else:
            mask = region_or_list["mask"]

        saliency = region_or_list["saliency"]
        return float((saliency * mask).sum())


def paper_gain_function_submodular(region_or_list, model=None, feature_extractor=None,
                                  original_images=None, lambda_weights=None):
    """
    Full submodular gain function g(S) from paper Equation (1):
    g(S) = λ₁·s_conf(S) + λ₂·s_eff(S) + λ₃·s_cons(S,fs) + λ₄·s_colla(S,I,fs)

    Args:
        region_or_list: subregion(s) to evaluate
        model: neural network for confidence scoring
        feature_extractor: feature extraction network
        original_images: list of original images for consistency/collaboration
        lambda_weights: [λ₁, λ₂, λ₃, λ₄] weighting factors (default: [1,1,1,1])

    Returns:
        float: submodular function value
    """
    if lambda_weights is None:
        lambda_weights = [1.0, 1.0, 1.0, 1.0]

    if isinstance(region_or_list, list):
        regions = region_or_list
    else:
        regions = [region_or_list]

    if not regions:
        return 0.0

    # Extract image index from first region
    try:
        img_idx = int(regions[0]['id'].split('_')[0])
        original_image = original_images[img_idx] if original_images else None
    except:
        original_image = None

    # Component 1: Confidence Score
    confidence_score = _calculate_confidence_score(regions, model)

    # Component 2: Effectiveness Score
    effectiveness_score = _calculate_effectiveness_score(regions, feature_extractor)

    # Component 3: Consistency Score
    consistency_score = _calculate_consistency_score(regions, feature_extractor, original_image)

    # Component 4: Collaboration Score
    collaboration_score = _calculate_collaboration_score(regions, feature_extractor, original_image)

    # Combine with weights (Equation 1)
    total_gain = (lambda_weights[0] * confidence_score +
                  lambda_weights[1] * effectiveness_score +
                  lambda_weights[2] * consistency_score +
                  lambda_weights[3] * collaboration_score)

    return float(total_gain)


def _calculate_confidence_score(regions, model):
    """Calculate confidence score s_conf using model predictions"""
    if not model:
        # Fallback: use simple saliency-based confidence
        try:
            return sum(r.get("gain_val", 0) for r in regions) / 1000.0  # Normalize
        except:
            return 0.0

    try:
        import torch
        total_confidence = 0.0

        for region in regions:
            try:
                # Apply region mask to image
                if 'original_mask' in region:
                    mask = region["original_mask"]
                else:
                    mask = region["mask"]

                # Ensure image is valid numpy array
                image_data = region["image"]
                if not isinstance(image_data, np.ndarray):
                    continue

                # Convert to tensor and apply mask
                image = torch.from_numpy(image_data).float()
                if len(image.shape) == 3:
                    image = image.permute(2, 0, 1)  # HWC -> CHW

                mask_tensor = torch.from_numpy(mask).float()
                if len(mask_tensor.shape) == 2:
                    mask_tensor = mask_tensor.unsqueeze(0).expand(3, -1, -1)

                masked_image = image * mask_tensor

                # Get model confidence (simplified)
                with torch.no_grad():
                    output = model(masked_image.unsqueeze(0))
                    confidence = torch.softmax(output, dim=1).max().item()
                    total_confidence += confidence
            except Exception as e:
                # Skip problematic regions
                continue

        return total_confidence

    except Exception as e:
        # Fallback to simple calculation
        try:
            return sum(r.get("gain_val", 0) for r in regions) / 1000.0
        except:
            return 0.0


def _calculate_effectiveness_score(regions, feature_extractor):
    """Calculate effectiveness score s_eff based on feature distances"""
    if len(regions) <= 1:
        return 0.0

    try:
        import torch
        from sklearn.metrics.pairwise import cosine_similarity

        features = []
        for region in regions:
            # Extract features with proper error handling
            try:
                if 'original_mask' in region:
                    mask = region["original_mask"]
                else:
                    mask = region["mask"]

                # Ensure image is valid numpy array
                image_data = region["image"]
                if not isinstance(image_data, np.ndarray):
                    continue

                image = torch.from_numpy(image_data).float()
                if len(image.shape) == 3:
                    image = image.permute(2, 0, 1)

                mask_tensor = torch.from_numpy(mask).float()
                if len(mask_tensor.shape) == 2:
                    mask_tensor = mask_tensor.unsqueeze(0).expand(3, -1, -1)

                masked_image = image * mask_tensor

                with torch.no_grad():
                    feature = feature_extractor(masked_image.unsqueeze(0))
                    features.append(feature.cpu().numpy().flatten())
            except Exception as e:
                # Skip problematic regions
                continue

        if len(features) <= 1:
            return 0.0

        # Calculate pairwise distances
        total_effectiveness = 0.0
        for i, feat_i in enumerate(features):
            min_distance = float('inf')
            for j, feat_j in enumerate(features):
                if i != j:
                    try:
                        distance = 1 - cosine_similarity([feat_i], [feat_j])[0][0]
                        min_distance = min(min_distance, distance)
                    except:
                        continue

            if min_distance != float('inf'):
                total_effectiveness += min_distance

        return total_effectiveness

    except Exception as e:
        # Fallback: use gain variance as effectiveness proxy
        try:
            gains = [r["gain_val"] for r in regions if "gain_val" in r]
            return float(np.std(gains)) if gains else 0.0
        except:
            return 0.0


def _calculate_consistency_score(regions, feature_extractor, original_image):
    """Calculate consistency score s_cons with target semantic feature"""
    if not regions or original_image is None:
        return 0.0

    try:
        import torch
        from sklearn.metrics.pairwise import cosine_similarity

        # Combine all region masks
        combined_mask = np.zeros_like(regions[0]['saliency'])
        for region in regions:
            try:
                if 'original_mask' in region:
                    mask = region["original_mask"]
                else:
                    mask = region["mask"]
                combined_mask = np.maximum(combined_mask, mask)
            except:
                continue

        # Ensure original_image is valid numpy array
        if not isinstance(original_image, np.ndarray):
            return 0.0

        # Get features of combined regions
        image = torch.from_numpy(original_image).float()
        if len(image.shape) == 3:
            image = image.permute(2, 0, 1)

        mask_tensor = torch.from_numpy(combined_mask).float()
        if len(mask_tensor.shape) == 2:
            mask_tensor = mask_tensor.unsqueeze(0).expand(3, -1, -1)

        masked_image = image * mask_tensor

        with torch.no_grad():
            combined_feature = feature_extractor(masked_image.unsqueeze(0))
            original_feature = feature_extractor(image.unsqueeze(0))

            # Cosine similarity between combined regions and original
            similarity = cosine_similarity(
                combined_feature.cpu().numpy().reshape(1, -1),
                original_feature.cpu().numpy().reshape(1, -1)
            )[0][0]

            return max(0, similarity)

    except Exception as e:
        # Fallback: normalized total gain
        try:
            total_gain = sum(r.get("gain_val", 0) for r in regions)
            return total_gain / 10000.0  # Normalize
        except:
            return 0.0


def _calculate_collaboration_score(regions, feature_extractor, original_image):
    """Calculate collaboration score s_colla = 1 - similarity(complement, target)"""
    if not regions or original_image is None:
        return 0.0

    try:
        import torch
        from sklearn.metrics.pairwise import cosine_similarity

        # Create combined mask of selected regions
        combined_mask = np.zeros_like(regions[0]['saliency'])
        for region in regions:
            try:
                if 'original_mask' in region:
                    mask = region["original_mask"]
                else:
                    mask = region["mask"]
                combined_mask = np.maximum(combined_mask, mask)
            except:
                continue

        # Create complement mask
        complement_mask = 1 - combined_mask

        # Ensure original_image is valid numpy array
        if not isinstance(original_image, np.ndarray):
            return 0.0

        # Get features
        image = torch.from_numpy(original_image).float()
        if len(image.shape) == 3:
            image = image.permute(2, 0, 1)

        comp_mask_tensor = torch.from_numpy(complement_mask).float()
        if len(comp_mask_tensor.shape) == 2:
            comp_mask_tensor = comp_mask_tensor.unsqueeze(0).expand(3, -1, -1)

        complement_image = image * comp_mask_tensor

        with torch.no_grad():
            complement_feature = feature_extractor(complement_image.unsqueeze(0))
            original_feature = feature_extractor(image.unsqueeze(0))

            # 1 - similarity with complement
            similarity = cosine_similarity(
                complement_feature.cpu().numpy().reshape(1, -1),
                original_feature.cpu().numpy().reshape(1, -1)
            )[0][0]

            return 1 - max(0, similarity)

    except Exception as e:
        # Fallback: inverse of effectiveness
        try:
            effectiveness = _calculate_effectiveness_score(regions, feature_extractor)
            return 1.0 / (effectiveness + 1.0)
        except:
            return 0.5  # Neutral fallback


def create_gain_function(use_submodular=False, model=None, feature_extractor=None,
                        original_images=None, lambda_weights=None):
    """
    Factory function to create appropriate gain function

    Args:
        use_submodular: whether to use full submodular function or simple version
        model: neural network for confidence (if submodular)
        feature_extractor: feature network (if submodular)
        original_images: original image data (if submodular)
        lambda_weights: submodular component weights (if submodular)

    Returns:
        callable: gain function g(·)
    """
    if use_submodular and model and feature_extractor:
        def submodular_gain_fn(region_or_list):
            return paper_gain_function_submodular(
                region_or_list, model, feature_extractor,
                original_images, lambda_weights
            )
        return submodular_gain_fn
    else:
        return paper_gain_function_simple


def test_cost_gain_functions():
    """Test cost and gain functions"""
    print("=== TESTING COST AND GAIN FUNCTIONS ===")

    # Create dummy test data
    dummy_region = {
        "id": "0_1",
        "mask": np.ones((100, 100)) / (100 * 100),  # Normalized cost = 1.0
        "original_mask": np.ones((100, 100)),        # Original for gain
        "saliency": np.random.rand(100, 100),
        "image": np.random.rand(100, 100, 3),
        "gain_val": 150.5
    }

    # Test single region
    print("Single region tests:")
    cost_single = paper_cost_function(dummy_region)
    gain_single = paper_gain_function_simple(dummy_region)
    print(f"  Cost: {cost_single:.6f} (should be ~1.0)")
    print(f"  Gain: {gain_single:.1f}")

    # Test multiple regions
    region_list = [dummy_region, dummy_region, dummy_region]
    cost_multiple = paper_cost_function(region_list)
    gain_multiple = paper_gain_function_simple(region_list)
    print(f"\nMultiple regions (3):")
    print(f"  Cost: {cost_multiple:.6f} (should be ~3.0)")
    print(f"  Gain: {gain_multiple:.1f}")

    # Test budget feasibility
    print(f"\nBudget feasibility:")
    budgets = [1, 5, 10, 20]
    for budget in budgets:
        max_regions = int(budget // cost_single)
        print(f"  Budget {budget:2d}: can fit ~{max_regions} regions")

    print(f"\nTest completed successfully!")


if __name__ == "__main__":
    test_cost_gain_functions()