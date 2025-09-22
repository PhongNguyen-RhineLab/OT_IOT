import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class SubmodularFunction:
    """
    Implementation of the submodular function from Equation (1) in the paper.
    g(S) = λ1*s_conf + λ2*s_eff + λ3*s_cons + λ4*s_colla
    """

    def __init__(self, model, feature_extractor, lambda1=1.0, lambda2=1.0,
                 lambda3=1.0, lambda4=1.0):
        self.model = model
        self.feature_extractor = feature_extractor
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.lambda4 = lambda4

    def confidence_score(self, regions):
        """
        Confidence score using EDL (Evidential Deep Learning)
        s_conf(X) for a set of regions
        """
        total_conf = 0
        for region in regions:
            # Apply region mask to get masked image
            masked_img = self.apply_mask(region['image'], region['mask'])

            # Get model prediction (simplified - should use EDL)
            with torch.no_grad():
                output = self.model(masked_img.unsqueeze(0))
                prob = F.softmax(output, dim=1)
                confidence = prob.max().item()  # Simplified confidence

            total_conf += confidence

        return total_conf

    def effectiveness_score(self, regions):
        """
        Effectiveness score: sum of minimum distances between region features
        s_eff(S) = Σ min_dist(F(si), F(sj)) for si in S
        """
        if len(regions) <= 1:
            return 0

        features = []
        for region in regions:
            masked_img = self.apply_mask(region['image'], region['mask'])
            with torch.no_grad():
                feature = self.feature_extractor(masked_img.unsqueeze(0)).detach().cpu().numpy()
            features.append(feature.flatten())

        total_eff = 0
        for i, feat_i in enumerate(features):
            min_dist = float('inf')
            for j, feat_j in enumerate(features):
                if i != j:
                    # Use cosine distance
                    dist = 1 - cosine_similarity([feat_i], [feat_j])[0][0]
                    min_dist = min(min_dist, dist)

            if min_dist != float('inf'):
                total_eff += min_dist

        return total_eff

    def consistency_score(self, regions, target_semantic_feature):
        """
        Consistency score: cosine similarity with target semantic
        s_cons(S, fs) = F(Σ I_M ∈ S) · fs / (||F(Σ I_M)|| ||fs||)
        """
        if not regions:
            return 0

        # Combine all masked regions
        combined_mask = np.zeros_like(regions[0]['mask'])
        for region in regions:
            combined_mask = np.maximum(combined_mask, region['mask'])

        # Apply combined mask
        masked_img = self.apply_mask(regions[0]['image'], combined_mask)
        with torch.no_grad():
            combined_feature = self.feature_extractor(masked_img.unsqueeze(0)).detach().cpu().numpy()

        # Cosine similarity
        similarity = cosine_similarity(
            combined_feature.reshape(1, -1),
            target_semantic_feature.reshape(1, -1)
        )[0][0]

        return max(0, similarity)  # Ensure non-negative

    def collaboration_score(self, regions, original_image, target_semantic_feature):
        """
        Collaboration score: 1 - similarity of (original - selected_regions) with target
        s_colla(S, I, fs) = 1 - F(I - Σ I_M) · fs / (||F(I - Σ I_M)|| ||fs||)
        """
        if not regions:
            return 0

        # Create combined mask for selected regions
        combined_mask = np.zeros_like(regions[0]['mask'])
        for region in regions:
            combined_mask = np.maximum(combined_mask, region['mask'])

        # Create complement (original - selected regions)
        complement_mask = 1 - combined_mask
        complement_img = self.apply_mask(original_image, complement_mask)

        with torch.no_grad():
            complement_feature = self.feature_extractor(complement_img.unsqueeze(0)).detach().cpu().numpy()

        # Cosine similarity
        similarity = cosine_similarity(
            complement_feature.reshape(1, -1),
            target_semantic_feature.reshape(1, -1)
        )[0][0]

        return 1 - max(0, similarity)

    def apply_mask(self, image, mask):
        """Apply mask to image"""
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).float()
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).float()

        # Ensure mask has same dimensions as image
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0).repeat(3, 1, 1)

        return image * mask

    def __call__(self, regions, original_image=None, target_semantic_feature=None):
        """
        Calculate submodular function value g(S)
        """
        if not regions:
            return 0

        # Calculate each component
        conf = self.confidence_score(regions)
        eff = self.effectiveness_score(regions)

        if target_semantic_feature is not None:
            cons = self.consistency_score(regions, target_semantic_feature)
            if original_image is not None:
                colla = self.collaboration_score(regions, original_image, target_semantic_feature)
            else:
                colla = 0
        else:
            cons = 0
            colla = 0

        # Combine with weights (Equation 1)
        total = (self.lambda1 * conf +
                 self.lambda2 * eff +
                 self.lambda3 * cons +
                 self.lambda4 * colla)

        return total


def create_gain_function(model, feature_extractor, original_images, **kwargs):
    """
    Factory function to create a gain function using the submodular function
    """
    submod_func = SubmodularFunction(model, feature_extractor, **kwargs)

    def gain_fn(region_or_set):
        if isinstance(region_or_set, list):
            regions = region_or_set
        else:
            regions = [region_or_set]

        # Get original image and target feature (simplified)
        if regions:
            img_id = int(regions[0]['id'].split('_')[0])
            original_img = torch.from_numpy(original_images[img_id]).float().permute(2, 0, 1)
            with torch.no_grad():
                target_feature = feature_extractor(original_img.unsqueeze(0)).detach().cpu().numpy()

            return submod_func(regions, original_img, target_feature)

        return 0

    return gain_fn