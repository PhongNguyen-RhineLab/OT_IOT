"""
dataset_loader.py - Dataset Loading and Saliency Map Generation
Paper: "Online approximate algorithms for Object detection under Budget allocation"
"""

import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100, STL10, MNIST, FashionMNIST
from torchvision.models import resnet18, ResNet18_Weights
import numpy as np


class DatasetLoader:
    """
    Load datasets and generate saliency maps for paper experiments
    """

    def __init__(self, model_name="resnet18", device="auto"):
        """
        Initialize dataset loader with model for saliency generation

        Args:
            model_name: Model to use for saliency ("resnet18")
            device: Device to use ("auto", "cpu", "cuda")
        """
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"Using device: {self.device}")

        # Load model for saliency generation
        if model_name == "resnet18":
            self.weights = ResNet18_Weights.DEFAULT
            self.model = resnet18(weights=self.weights)
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        self.model.to(self.device)
        self.model.eval()

        # Setup transforms
        self.transform = self._get_transforms()

    def _get_transforms(self):
        """Get standard transforms for the model"""
        # Use model's default transforms if available
        try:
            return self.weights.transforms()
        except:
            # Fallback to manual transforms
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

    def load_dataset(self, dataset_name, num_samples=20, data_root="./data"):
        """
        Load dataset and generate saliency maps

        Args:
            dataset_name: Name of dataset ("cifar10", "cifar100", etc.)
            num_samples: Number of samples to load
            data_root: Root directory for dataset storage

        Returns:
            tuple: (images, saliency_maps) as numpy arrays
        """
        print(f"Loading {dataset_name} dataset ({num_samples} samples)...")

        # Load dataset
        dataset = self._load_torch_dataset(dataset_name, data_root)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

        # Generate images and saliency maps
        images = []
        saliency_maps = []

        for i, (img_tensor, label) in enumerate(dataloader):
            if i >= num_samples:
                break

            # Convert tensor to numpy for processing (H,W,C format)
            img_np = self._tensor_to_numpy(img_tensor[0])
            images.append(img_np)

            # Generate saliency map
            saliency = self._generate_saliency_map(img_tensor, label)
            saliency_maps.append(saliency)

            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{num_samples} samples")

        print(f"Dataset loading completed: {len(images)} images, {len(saliency_maps)} saliency maps")
        return images, saliency_maps

    def _load_torch_dataset(self, dataset_name, data_root):
        """Load PyTorch dataset"""
        datasets = {
            "cifar10": lambda: CIFAR10(root=data_root, train=False, download=True, transform=self.transform),
            "cifar100": lambda: CIFAR100(root=data_root, train=False, download=True, transform=self.transform),
            "stl10": lambda: STL10(root=data_root, split="test", download=True, transform=self.transform),
            "mnist": lambda: MNIST(root=data_root, train=False, download=True,
                                   transform=transforms.Compose([
                                       transforms.Grayscale(num_output_channels=3),
                                       *self.transform.transforms
                                   ])),
            "fashionmnist": lambda: FashionMNIST(root=data_root, train=False, download=True,
                                                 transform=transforms.Compose([
                                                     transforms.Grayscale(num_output_channels=3),
                                                     *self.transform.transforms
                                                 ]))
        }

        if dataset_name not in datasets:
            raise ValueError(f"Unsupported dataset: {dataset_name}. Available: {list(datasets.keys())}")

        return datasets[dataset_name]()

    def _tensor_to_numpy(self, tensor):
        """Convert tensor to numpy array in (H,W,C) format"""
        # tensor is (C,H,W), convert to (H,W,C)
        if len(tensor.shape) == 3:
            numpy_img = tensor.permute(1, 2, 0).numpy()
        else:
            numpy_img = tensor.numpy()

        # Denormalize if needed (approximate)
        if numpy_img.min() < 0:
            # Approximate denormalization
            numpy_img = (numpy_img - numpy_img.min()) / (numpy_img.max() - numpy_img.min())

        return numpy_img.astype(np.float32)

    def _generate_saliency_map(self, img_tensor, label):
        """
        Generate saliency map using gradient-based method

        Args:
            img_tensor: Input image tensor (1,C,H,W)
            label: Ground truth label (not used in current implementation)

        Returns:
            numpy array: Saliency map (H,W)
        """
        try:
            return self._gradcam_saliency(img_tensor)
        except Exception as e:
            print(f"Warning: GradCAM failed ({e}), using simple gradient method")
            return self._simple_gradient_saliency(img_tensor)

    def _gradcam_saliency(self, img_tensor):
        """Generate saliency using GradCAM-like method"""
        img_tensor = img_tensor.to(self.device)
        img_tensor.requires_grad_(True)

        # Forward pass
        output = self.model(img_tensor)
        pred_class = output.argmax(dim=1).item()

        # Backward pass
        self.model.zero_grad()
        class_score = output[0, pred_class]
        class_score.backward()

        # Get gradients and activations (simplified)
        gradients = img_tensor.grad.detach()

        # Create saliency map (simplified version)
        saliency = torch.abs(gradients).mean(dim=1).squeeze()  # Average over channels
        saliency = torch.nn.functional.interpolate(
            saliency.unsqueeze(0).unsqueeze(0),
            size=(224, 224),
            mode='bilinear'
        ).squeeze()

        # Normalize to [0,1]
        saliency_np = saliency.cpu().numpy()
        if saliency_np.max() > saliency_np.min():
            saliency_np = (saliency_np - saliency_np.min()) / (saliency_np.max() - saliency_np.min())

        return saliency_np.astype(np.float32)

    def _simple_gradient_saliency(self, img_tensor):
        """Fallback: simple gradient-based saliency"""
        img_tensor = img_tensor.to(self.device)
        img_tensor.requires_grad_(True)

        # Forward pass
        output = self.model(img_tensor)
        pred_class = output.argmax(dim=1).item()

        # Backward pass
        self.model.zero_grad()
        output[0, pred_class].backward()

        # Simple gradient magnitude
        gradients = img_tensor.grad.detach().abs().mean(dim=1).squeeze()

        # Convert to numpy and normalize
        saliency_np = gradients.cpu().numpy()
        if saliency_np.max() > saliency_np.min():
            saliency_np = (saliency_np - saliency_np.min()) / (saliency_np.max() - saliency_np.min())

        return saliency_np.astype(np.float32)

    def get_model_info(self):
        """Get model information for submodular functions"""
        return {
            'model': self.model,
            'feature_extractor': torch.nn.Sequential(*list(self.model.children())[:-1]),
            'device': self.device,
            'weights': self.weights
        }


def test_dataset_loader():
    """Test dataset loading functionality"""
    print("=== TESTING DATASET LOADER ===")

    loader = DatasetLoader()

    # Test with small sample
    datasets_to_test = ["cifar10"]
    num_samples = 5

    for dataset_name in datasets_to_test:
        print(f"\nTesting {dataset_name}...")
        try:
            images, saliency_maps = loader.load_dataset(dataset_name, num_samples)

            # Validate results
            print(f"  Images: {len(images)} samples")
            print(f"  Saliency maps: {len(saliency_maps)} samples")

            if images:
                print(f"  Sample image shape: {images[0].shape}")
                print(f"  Sample image range: [{images[0].min():.3f}, {images[0].max():.3f}]")

            if saliency_maps:
                print(f"  Sample saliency shape: {saliency_maps[0].shape}")
                print(f"  Sample saliency range: [{saliency_maps[0].min():.3f}, {saliency_maps[0].max():.3f}]")

        except Exception as e:
            print(f"  ERROR loading {dataset_name}: {e}")
            import traceback
            traceback.print_exc()

    print("\nDataset loader test completed!")


if __name__ == "__main__":
    test_dataset_loader()