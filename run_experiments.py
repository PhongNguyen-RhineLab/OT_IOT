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

from tracked_algorithms import Greedy_Search_Tracked, OT_Algorithm_Tracked, IOT_Algorithm_Tracked
from gradcam import gradcam
from submodular_function import create_gain_function


# Enhanced benchmark with operation tracking
def benchmark_with_tracking(func, *args, **kwargs):
    """Benchmark with runtime, memory and operation tracking"""
    tracemalloc.start()
    start = time.time()

    result, aux_data = func(*args, **kwargs)

    runtime = time.time() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Extract operation summary
    operation_summary = aux_data.get('operation_summary', {})

    return result, runtime, peak / 1024, operation_summary, aux_data


# ----------------- Memory calculation helper ----------------- #
def calculate_sizeof_subregion(sample_region):
    """Calculate size of one subregion in KB"""
    size_bytes = 0

    if 'mask' in sample_region:
        mask = sample_region['mask']
        size_bytes += mask.nbytes

    if 'saliency' in sample_region:
        saliency = sample_region['saliency']
        size_bytes += saliency.nbytes

    if 'image' in sample_region:
        image = sample_region['image']
        if isinstance(image, np.ndarray):
            size_bytes += image.nbytes
        else:
            size_bytes += 224 * 224 * 3 * 4

    size_bytes += 64
    return size_bytes / 1024


def calculate_theoretical_memory(algorithm, num_images, m_per_image, solution_set,
                                 sample_subregion, aux_data=None):
    """Calculate theoretical memory usage"""
    sizeof_subregion = calculate_sizeof_subregion(sample_subregion)

    if algorithm == "Greedy":
        memory = (num_images * m_per_image + len(solution_set)) * sizeof_subregion

    elif algorithm == "OT":
        S_size = aux_data.get('S_size', 0) if aux_data else 0
        S_prime_size = aux_data.get('S_prime_size', 0) if aux_data else 0
        I_star_size = 1 if aux_data and aux_data.get('has_I_star', False) else 0

        memory = m_per_image * sizeof_subregion + (S_size + S_prime_size + I_star_size) * sizeof_subregion

    elif algorithm == "IOT":
        S_size = aux_data.get('S_size', 0) if aux_data else 0
        S_prime_size = aux_data.get('S_prime_size', 0) if aux_data else 0
        I_star_size = 1 if aux_data and aux_data.get('has_I_star', False) else 0

        max_S_tau = aux_data.get('max_S_tau_size', 0) if aux_data else 0
        max_S_prime_tau = aux_data.get('max_S_prime_tau_size', 0) if aux_data else 0
        S_b_size = aux_data.get('S_b_size', 0) if aux_data else 0

        memory = (m_per_image * sizeof_subregion +
                  (S_size + S_prime_size + I_star_size) * sizeof_subregion +
                  (max_S_tau + max_S_prime_tau + S_b_size) * sizeof_subregion)
    else:
        memory = 0

    return memory


def benchmark_with_theoretical_memory(func, algorithm, num_images, m_per_image, *args, **kwargs):
    """Benchmark with both runtime, operation tracking and theoretical memory calculation"""
    start = time.time()
    result, aux_memory_data = func(*args, **kwargs)
    runtime = time.time() - start

    # Extract operation summary
    operation_summary = aux_memory_data.get('operation_summary', {})

    # Extract solution and sample subregion for memory calculation
    if result and len(result) >= 2 and isinstance(result[0], list):
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

    return result, runtime, theoretical_memory, operation_summary, aux_memory_data


# ----------------- Cost & Gain ----------------- #
def cost_fn(region_or_list):
    """Handle both single region and list of regions"""
    if isinstance(region_or_list, list):
        if not region_or_list:
            return 0
        total_cost = 0
        for region in region_or_list:
            total_cost += region["mask"].sum()
        return total_cost
    else:
        return region_or_list["mask"].sum()


def simple_gain_fn(region_or_list):
    """Handle both single region and list of regions"""
    if isinstance(region_or_list, list):
        if not region_or_list:
            return 0
        total_gain = 0
        for region in region_or_list:
            total_gain += (region["saliency"] * region["mask"]).sum()
        return total_gain
    else:
        return (region_or_list["saliency"] * region_or_list["mask"]).sum()


def build_base_transform(weights):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    try:
        meta = getattr(weights, "meta", {})
        mean = meta.get("mean", mean)
        std = meta.get("std", std)
    except Exception:
        pass
    return mean, std


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

    # Choose gain function (keep original logic)
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
    operation_results = []

    for B in args.budgets:
        print(f"\n=== Budget: {B} ===")

        # Greedy Search
        print("Running Greedy Search...")
        (S_gs, g_gs), t_gs, mem_gs, ops_gs, aux_gs = benchmark_with_theoretical_memory(
            Greedy_Search_Tracked, "Greedy", args.num_samples, args.m,
            images, saliency_maps, N=4, m=args.m,
            budget=B, cost_fn=cost_fn, gain_fn=gain_fn
        )
        results.append(["Greedy", args.dataset, B, "-", len(S_gs), g_gs, t_gs, mem_gs])
        operation_results.append(["Greedy", B, ops_gs.get('total_oracle_calls', 0),
                                  ops_gs.get('gain_calls', 0), ops_gs.get('marginal_gain_calls', 0),
                                  ops_gs.get('threshold_checks', 0), ops_gs.get('iterations', 0)])
        print(f"Greedy: |S|={len(S_gs)}, g(S)={g_gs:.3f}, time={t_gs:.3f}s, mem={mem_gs:.1f}KB")
        print(f"        Oracle calls: {ops_gs.get('total_oracle_calls', 0)}")

        # OT Algorithm
        print("Running OT Algorithm...")
        (S_ot, g_ot), t_ot, mem_ot, ops_ot, aux_ot = benchmark_with_theoretical_memory(
            OT_Algorithm_Tracked, "OT", args.num_samples, args.m,
            images, saliency_maps, N=4, m=args.m,
            budget=B, cost_fn=cost_fn, gain_fn=gain_fn
        )
        results.append(["OT", args.dataset, B, "-", len(S_ot), g_ot, t_ot, mem_ot])
        operation_results.append(["OT", B, ops_ot.get('total_oracle_calls', 0),
                                  ops_ot.get('gain_calls', 0), ops_ot.get('marginal_gain_calls', 0),
                                  ops_ot.get('threshold_checks', 0), ops_ot.get('iterations', 0)])
        print(f"OT: |S|={len(S_ot)}, g(S)={g_ot:.3f}, time={t_ot:.3f}s, mem={mem_ot:.1f}KB")
        print(f"    Oracle calls: {ops_ot.get('total_oracle_calls', 0)}")

        # IOT Algorithm
        for eps in args.epsilons:
            print(f"Running IOT Algorithm (ε={eps})...")
            (S_iot, g_iot), t_iot, mem_iot, ops_iot, aux_iot = benchmark_with_theoretical_memory(
                IOT_Algorithm_Tracked, "IOT", args.num_samples, args.m,
                images, saliency_maps, N=4, m=args.m,
                budget=B, eps=eps, cost_fn=cost_fn, gain_fn=gain_fn
            )
            results.append(["IOT", args.dataset, B, eps, len(S_iot), g_iot, t_iot, mem_iot])
            operation_results.append(["IOT", B, ops_iot.get('total_oracle_calls', 0),
                                      ops_iot.get('gain_calls', 0), ops_iot.get('marginal_gain_calls', 0),
                                      ops_iot.get('threshold_checks', 0), ops_iot.get('iterations', 0)])
            print(f"IOT (ε={eps}): |S|={len(S_iot)}, g(S)={g_iot:.3f}, time={t_iot:.3f}s, mem={mem_iot:.1f}KB")
            print(f"               Oracle calls: {ops_iot.get('total_oracle_calls', 0)}")

    # Create results DataFrames
    df_results = pd.DataFrame(results, columns=[
        "Algorithm", "Dataset", "Budget", "Epsilon", "SetSize", "Gain", "Time(s)", "Memory(KB)"
    ])

    df_operations = pd.DataFrame(operation_results, columns=[
        "Algorithm", "Budget", "Total_Oracle_Calls", "Gain_Calls", "Marginal_Gain_Calls",
        "Threshold_Checks", "Iterations"
    ])

    # Save results
    suffix = "_submodular" if args.use_submodular else "_simple"
    results_file = f"experiment_results{suffix}.csv"
    operations_file = f"operation_analysis{suffix}.csv"

    df_results.to_csv(results_file, index=False)
    df_operations.to_csv(operations_file, index=False)

    print(f"\nSaved results to {results_file}")
    print(f"Saved operation analysis to {operations_file}")

    # Display final results
    print(f"\n=== FINAL RESULTS ===")
    print(df_results)

    print(f"\n=== OPERATION ANALYSIS ===")
    print(df_operations)

    # Theoretical vs Actual Complexity Analysis
    print(f"\n=== COMPLEXITY ANALYSIS ===")
    n = args.num_samples * args.m  # Total regions

    print(f"Total regions (n): {n}")
    print(f"Algorithm    | Actual Oracle Calls | Theoretical | Ratio")
    print(f"-------------|-------------------|-------------|-------")

    for _, row in df_operations.iterrows():
        alg = row['Algorithm']
        oracle_calls = row['Total_Oracle_Calls']

        if alg == "Greedy":
            theoretical = n ** 2  # O(n^2) worst case
            theoretical_str = f"O(n²) ≈ {theoretical}"
        elif alg == "OT":
            theoretical = 3 * n  # 3n as per paper
            theoretical_str = f"3n ≈ {theoretical}"
        elif alg == "IOT":
            theoretical = 5 * n  # 5n as per paper
            theoretical_str = f"5n ≈ {theoretical}"
        else:
            theoretical = 0
            theoretical_str = "Unknown"

        ratio = oracle_calls / theoretical if theoretical > 0 else 0
        print(f"{alg:12s} | {oracle_calls:17d} | {theoretical_str:11s} | {ratio:.2f}")