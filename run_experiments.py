import argparse
import time
import tracemalloc
import numpy as np
import pandas as pd
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.datasets import (
    CIFAR10, CIFAR100, STL10,
    MNIST, FashionMNIST, ImageFolder
)

from greedy_search import Greedy_Search
from ot_algo import OT_algorithm
from iot_algo import IOT_algorithm
from gradcam import gradcam
from submodular_function import create_gain_function


# ----------------- Memory calculation helper ----------------- #
def calculate_sizeof_subregion(sample_region):
    """Calculate size of one subregion in KB"""
    size_bytes = 0

    # Size of mask (H × W × 4 bytes for float32)
    if 'mask' in sample_region:
        mask = sample_region['mask']
        size_bytes += mask.nbytes

    # Size of saliency map (H × W × 4 bytes for float32)
    if 'saliency' in sample_region:
        saliency = sample_region['saliency']
        size_bytes += saliency.nbytes

    # Size of image reference (H × W × C × 4 bytes for float32)
    if 'image' in sample_region:
        image = sample_region['image']
        if isinstance(image, np.ndarray):
            size_bytes += image.nbytes
        else:
            # Estimate size if not numpy array
            size_bytes += 224 * 224 * 3 * 4  # Typical CNN input size

    # Size of metadata (id, gain_val, etc.) - small overhead
    size_bytes += 64  # Estimated string + float overhead

    return size_bytes / 1024  # Convert to KB


def calculate_theoretical_memory(algorithm, num_images, m, solution_set,
                                 sample_subregion, aux_data=None):
    """
    Calculate theoretical memory usage according to the formulas:

    Greedy: M = Số ảnh * m * sizeof(1 subregion) + |S| * sizeof(1 subregion)
    OT: M = m * sizeof(1 sub) + |S+S'+1| * sizeof(1 sub)
    IOT: M = m * sizeof(1 sub) + |S+S'+1| * sizeof(1 sub) + |max of S_tau + max of S'_tau + S_b| * sizeof(1 sub)
    """
    sizeof_subregion = calculate_sizeof_subregion(sample_subregion)

    if algorithm == "Greedy":
        # M = Số ảnh * m * sizeof(1 subregion) + |S| * sizeof(1 subregion)
        memory = (num_images * m + len(solution_set)) * sizeof_subregion

    elif algorithm == "OT":
        # M = m * sizeof(1 sub) + |S+S'+1| * sizeof(1 sub)
        # aux_data should contain sizes of S, S', and I_star
        S_size = aux_data.get('S_size', 0) if aux_data else 0
        S_prime_size = aux_data.get('S_prime_size', 0) if aux_data else 0
        I_star_size = 1 if aux_data and aux_data.get('has_I_star', False) else 0

        memory = m * sizeof_subregion + (S_size + S_prime_size + I_star_size) * sizeof_subregion

    elif algorithm == "IOT":
        # M = m * sizeof(1 sub) + |S+S'+1| * sizeof(1 sub) + |max of S_tau + max of S'_tau + S_b| * sizeof(1 sub)
        S_size = aux_data.get('S_size', 0) if aux_data else 0
        S_prime_size = aux_data.get('S_prime_size', 0) if aux_data else 0
        I_star_size = 1 if aux_data and aux_data.get('has_I_star', False) else 0

        # Additional memory for IOT candidates
        max_S_tau = aux_data.get('max_S_tau_size', 0) if aux_data else 0
        max_S_prime_tau = aux_data.get('max_S_prime_tau_size', 0) if aux_data else 0
        S_b_size = aux_data.get('S_b_size', 0) if aux_data else 0

        memory = (m * sizeof_subregion +
                  (S_size + S_prime_size + I_star_size) * sizeof_subregion +
                  (max_S_tau + max_S_prime_tau + S_b_size) * sizeof_subregion)
    else:
        memory = 0

    return memory


# ----------------- Benchmark helper with theoretical memory ----------------- #
def benchmark_with_theoretical_memory(func, algorithm, num_images, m_per_image, *args, **kwargs):
    """Benchmark with both runtime and theoretical memory calculation"""
    start = time.time()
    result, aux_memory_data = func(*args, **kwargs)
    runtime = time.time() - start

    # Extract solution and sample subregion for memory calculation
    if result and len(result) > 0 and isinstance(result[0], list):
        solution_set = result[0]
        sample_subregion = solution_set[0] if solution_set else None
    else:
        solution_set = []
        sample_subregion = None

    # Calculate theoretical memory if we have sample
    if sample_subregion:
        theoretical_memory = calculate_theoretical_memory(
            algorithm, num_images, m_per_image, solution_set, sample_subregion, aux_memory_data
        )
    else:
        theoretical_memory = 0

    return result, runtime, theoretical_memory


# ----------------- Cost & Gain ----------------- #
def cost_fn(region):
    return region["mask"].sum()


# Simple gain function - keep as backup
def simple_gain_fn(region):
    return (region["saliency"] * region["mask"]).sum()


def build_base_transform(weights):
    # Robust extraction of mean/std
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    try:
        meta = getattr(weights, "meta", {})
        mean = meta.get("mean", mean)
        std = meta.get("std", std)
    except Exception:
        pass
    return mean, std


# ----------------- Dataset loader ----------------- #
def load_dataset(name: str, weights, root: str, num_samples: int):
    mean, std = build_base_transform(weights)

    def pipeline(extra=None):
        ops = [transforms.Resize((224, 224))]
        if extra:
            ops.extend(extra)
        ops += [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
        return transforms.Compose(ops)

    if name == "cifar10":
        ds = CIFAR10(root=root, train=False, download=True, transform=pipeline())
    elif name == "cifar100":
        ds = CIFAR100(root=root, train=False, download=True, transform=pipeline())
    elif name == "stl10":
        ds = STL10(root=root, split="test", download=True, transform=pipeline())
    elif name == "mnist":
        ds = MNIST(root=root, train=False, download=True,
                   transform=pipeline([transforms.Grayscale(num_output_channels=3)]))
    elif name == "fashionmnist":
        ds = FashionMNIST(root=root, train=False, download=True,
                          transform=pipeline([transforms.Grayscale(num_output_channels=3)]))
    elif name == "imagenet":
        if not root:
            raise ValueError("Provide --data-root pointing to ImageNet split directory.")
        try:
            tf = weights.transforms()
            if not hasattr(tf, "__call__"):
                raise RuntimeError
            ds = ImageFolder(root=root, transform=tf)
        except Exception:
            ds = ImageFolder(root=root, transform=pipeline())
    else:
        raise ValueError(f"Unsupported dataset: {name}")

    loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=True)
    images, saliency_maps = [], []
    for img, _ in loader:
        cam = gradcam(model, img)
        images.append(img[0].permute(1, 2, 0).numpy())
        saliency_maps.append(cam)
        if len(images) >= num_samples:
            break
    return images, saliency_maps


# ----------------- Main ----------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cifar10",
                        choices=["cifar10", "cifar100", "stl10",
                                 "mnist", "fashionmnist", "imagenet"])
    parser.add_argument("--data-root", default="./data",
                        help="Root folder (used for downloads or ImageNet path).")
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--budgets", type=int, nargs="+", default=[2000, 4000, 8000])
    parser.add_argument("--epsilons", type=float, nargs="+", default=[0.1, 0.3])
    parser.add_argument("--use-submodular", action="store_true",
                        help="Use full submodular function instead of simple saliency")
    parser.add_argument("--m", type=int, default=5, help="Number of subregions per image")
    args = parser.parse_args()

    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.eval()

    images, saliency_maps = load_dataset(
        args.dataset, weights, args.data_root, args.num_samples
    )

    # Choose gain function
    if args.use_submodular:
        print("Using full submodular function (4 components)")
        feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
        feature_extractor.eval()

        gain_fn = create_gain_function(
            model=model,
            feature_extractor=feature_extractor,
            original_images=images,
            lambda1=1.0, lambda2=1.0, lambda3=1.0, lambda4=1.0
        )
    else:
        print("Using simple saliency-based gain function")
        gain_fn = simple_gain_fn

    results = []
    for B in args.budgets:
        print(f"\n=== Budget: {B} ===")

        # Greedy Search
        (S_gs, g_gs), t_gs, mem_gs = benchmark_with_theoretical_memory(
            Greedy_Search, "Greedy", args.num_samples, args.m,
            images, saliency_maps, N=4, m=args.m,
            budget=B, cost_fn=cost_fn, gain_fn=gain_fn
        )
        results.append(["Greedy", args.dataset, B, "-", len(S_gs), g_gs, t_gs, mem_gs])
        print(f"Greedy: |S|={len(S_gs)}, g(S)={g_gs:.3f}, time={t_gs:.3f}s, mem={mem_gs:.1f}KB")

        # OT Algorithm
        (S_ot, g_ot), t_ot, mem_ot = benchmark_with_theoretical_memory(
            OT_algorithm, "OT", args.num_samples, args.m,
            images, saliency_maps, N=4, m=args.m,
            budget=B, cost_fn=cost_fn, gain_fn=gain_fn
        )
        results.append(["OT", args.dataset, B, "-", len(S_ot), g_ot, t_ot, mem_ot])
        print(f"OT: |S|={len(S_ot)}, g(S)={g_ot:.3f}, time={t_ot:.3f}s, mem={mem_ot:.1f}KB")

        # IOT Algorithm
        for eps in args.epsilons:
            (S_iot, g_iot), t_iot, mem_iot = benchmark_with_theoretical_memory(
                IOT_algorithm, "IOT", args.num_samples, args.m,
                images, saliency_maps, N=4, m=args.m,
                budget=B, eps=eps, cost_fn=cost_fn, gain_fn=gain_fn
            )
            results.append(["IOT", args.dataset, B, eps, len(S_iot), g_iot, t_iot, mem_iot])
            print(f"IOT (ε={eps}): |S|={len(S_iot)}, g(S)={g_iot:.3f}, time={t_iot:.3f}s, mem={mem_iot:.1f}KB")

    df = pd.DataFrame(results, columns=[
        "Algorithm", "Dataset", "Budget", "Epsilon", "SetSize", "Gain", "Time(s)", "Memory(KB)"
    ])

    suffix = "_submodular" if args.use_submodular else "_simple"
    out_file = f"experiment_results{suffix}.csv"
    df.to_csv(out_file, index=False)
    print(f"\nSaved results to {out_file}")
    print(df)