import argparse
import time
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.datasets import (
    CIFAR10, CIFAR100, STL10,
    MNIST, FashionMNIST, ImageFolder
)

from tracked_algorithms import Greedy_Search_Tracked, OT_Algorithm_Tracked, IOT_Algorithm_Tracked
from gradcam import gradcam
from submodular_function import create_gain_function


# ----------------- Theoretical Memory Calculation ----------------- #
def calculate_sizeof_subregion(sample_region):
    """Calculate size of one subregion in KB according to data structure"""
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
            # Estimate standard size
            size_bytes += 224 * 224 * 3 * 4

    # Metadata overhead (id string, gain_val float, etc.)
    size_bytes += 64

    return size_bytes / 1024  # Convert to KB


def calculate_theoretical_memory(algorithm, num_images, m_per_image, solution_set,
                                 sample_subregion, aux_data=None):
    """
    Calculate theoretical memory according to formulas discussed:

    Greedy: M = Số ảnh * m * sizeof(1 subregion) + |S| * sizeof(1 subregion)
    OT: M = m * sizeof(1 sub) + |S+S'+1| * sizeof(1 sub)
    IOT: M = m * sizeof(1 sub) + |S+S'+1| * sizeof(1 sub) + |max of S_tau + max of S'_tau + S_b| * sizeof(1 sub)
    """
    if not sample_subregion:
        return 0

    sizeof_subregion = calculate_sizeof_subregion(sample_subregion)

    if algorithm == "Greedy":
        # M = Số ảnh * m * sizeof(1 subregion) + |S| * sizeof(1 subregion)
        memory = (num_images * m_per_image + len(solution_set)) * sizeof_subregion

    elif algorithm == "OT":
        # M = m * sizeof(1 sub) + |S+S'+1| * sizeof(1 sub)
        S_size = aux_data.get('S_size', 0) if aux_data else 0
        S_prime_size = aux_data.get('S_prime_size', 0) if aux_data else 0
        I_star_size = 1 if aux_data and aux_data.get('has_I_star', False) else 0

        memory = m_per_image * sizeof_subregion + (S_size + S_prime_size + I_star_size) * sizeof_subregion

    elif algorithm == "IOT":
        # M = m * sizeof(1 sub) + |S+S'+1| * sizeof(1 sub) + |max of S_tau + max of S'_tau + S_b| * sizeof(1 sub)
        S_size = aux_data.get('S_size', 0) if aux_data else 0
        S_prime_size = aux_data.get('S_prime_size', 0) if aux_data else 0
        I_star_size = 1 if aux_data and aux_data.get('has_I_star', False) else 0

        # Additional IOT-specific memory components
        max_S_tau = aux_data.get('max_S_tau_size', 0) if aux_data else 0
        max_S_prime_tau = aux_data.get('max_S_prime_tau_size', 0) if aux_data else 0
        S_b_size = aux_data.get('S_b_size', 0) if aux_data else 0

        # Base memory + dual candidates + IOT candidates
        base_memory = m_per_image * sizeof_subregion
        dual_candidates_memory = (S_size + S_prime_size + I_star_size) * sizeof_subregion
        iot_specific_memory = (max_S_tau + max_S_prime_tau + S_b_size) * sizeof_subregion

        memory = base_memory + dual_candidates_memory + iot_specific_memory

    else:
        memory = 0

    return memory


def benchmark_with_theoretical_memory(func, algorithm, num_images, m_per_image, *args, **kwargs):
    """Benchmark with runtime and theoretical memory calculation"""
    start = time.time()
    result, aux_memory_data = func(*args, **kwargs)
    runtime = time.time() - start

    # Extract operation summary
    operation_summary = aux_memory_data.get('operation_summary', {})

    # Extract solution set for memory calculation
    if result and len(result) >= 2:
        solution_set = result[0]
        sample_subregion = solution_set[0] if solution_set else None
    else:
        solution_set = []
        sample_subregion = None

    # Calculate theoretical memory
    theoretical_memory = calculate_theoretical_memory(
        algorithm, num_images, m_per_image, solution_set, sample_subregion, aux_memory_data
    )

    return result, runtime, theoretical_memory, operation_summary, aux_memory_data


# ----------------- Cost & Gain Functions ----------------- #
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


# ----------------- Main Experiment Runner ----------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cifar10",
                        choices=["cifar10", "cifar100", "stl10",
                                 "mnist", "fashionmnist", "imagenet"])
    parser.add_argument("--data-root", default="./data")
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

    print(f"Loading {args.num_samples} samples from {args.dataset}...")
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
    operation_results = []
    gain_breakdown_data = []
    memory_breakdown_data = []

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
                                  ops_gs.get('total_gain_calls', 0), ops_gs.get('marginal_gain_calls', 0),
                                  ops_gs.get('threshold_checks', 0), ops_gs.get('iterations', 0)])
        gain_breakdown_data.append(
            ["Greedy", B, ops_gs.get('total_gain_calls', 0), ops_gs.get('singleton_gain_calls', 0),
             ops_gs.get('marginal_gain_calls', 0), ops_gs.get('gain_union_calls', 0),
             ops_gs.get('gain_current_set_calls', 0)])

        # Memory breakdown for Greedy
        sizeof_subregion = calculate_sizeof_subregion(S_gs[0]) if S_gs else 0
        greedy_theoretical = (args.num_samples * args.m + len(S_gs)) * sizeof_subregion
        memory_breakdown_data.append(
            ["Greedy", B, sizeof_subregion, args.num_samples * args.m, len(S_gs), 0, 0, 0, greedy_theoretical])

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
                                  ops_ot.get('total_gain_calls', 0), ops_ot.get('marginal_gain_calls', 0),
                                  ops_ot.get('threshold_checks', 0), ops_ot.get('iterations', 0)])
        gain_breakdown_data.append(["OT", B, ops_ot.get('total_gain_calls', 0), ops_ot.get('singleton_gain_calls', 0),
                                    ops_ot.get('marginal_gain_calls', 0), ops_ot.get('gain_union_calls', 0),
                                    ops_ot.get('gain_current_set_calls', 0)])

        # Memory breakdown for OT: M = m * sizeof(1 sub) + |S+S'+1| * sizeof(1 sub)
        S_size = aux_ot.get('S_size', 0)
        S_prime_size = aux_ot.get('S_prime_size', 0)
        I_star_size = 1 if aux_ot.get('has_I_star', False) else 0
        ot_theoretical = args.m * sizeof_subregion + (S_size + S_prime_size + I_star_size) * sizeof_subregion
        memory_breakdown_data.append(
            ["OT", B, sizeof_subregion, args.m, S_size + S_prime_size + I_star_size, 0, 0, 0, ot_theoretical])

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
                                      ops_iot.get('total_gain_calls', 0), ops_iot.get('marginal_gain_calls', 0),
                                      ops_iot.get('threshold_checks', 0), ops_iot.get('iterations', 0)])
            gain_breakdown_data.append(
                ["IOT", B, ops_iot.get('total_gain_calls', 0), ops_iot.get('singleton_gain_calls', 0),
                 ops_iot.get('marginal_gain_calls', 0), ops_iot.get('gain_union_calls', 0),
                 ops_iot.get('gain_current_set_calls', 0)])

            # Memory breakdown for IOT
            iot_S_size = aux_iot.get('S_size', 0)
            iot_S_prime_size = aux_iot.get('S_prime_size', 0)
            iot_I_star_size = 1 if aux_iot.get('has_I_star', False) else 0
            max_S_tau = aux_iot.get('max_S_tau_size', 0)
            max_S_prime_tau = aux_iot.get('max_S_prime_tau_size', 0)
            S_b_size = aux_iot.get('S_b_size', 0)

            base_mem = args.m * sizeof_subregion
            dual_mem = (iot_S_size + iot_S_prime_size + iot_I_star_size) * sizeof_subregion
            iot_specific_mem = (max_S_tau + max_S_prime_tau + S_b_size) * sizeof_subregion
            iot_theoretical = base_mem + dual_mem + iot_specific_mem

            memory_breakdown_data.append(
                ["IOT", B, sizeof_subregion, args.m, iot_S_size + iot_S_prime_size + iot_I_star_size,
                 max_S_tau + max_S_prime_tau + S_b_size, base_mem, dual_mem, iot_theoretical])

            print(f"IOT (ε={eps}): |S|={len(S_iot)}, g(S)={g_iot:.3f}, time={t_iot:.3f}s, mem={mem_iot:.1f}KB")
            print(f"               Oracle calls: {ops_iot.get('total_oracle_calls', 0)}")

    # Create DataFrames
    df_results = pd.DataFrame(results, columns=[
        "Algorithm", "Dataset", "Budget", "Epsilon", "SetSize", "Gain", "Time(s)", "Memory(KB)"
    ])

    df_operations = pd.DataFrame(operation_results, columns=[
        "Algorithm", "Budget", "Total_Oracle_Calls", "Total_Gain_Calls", "Marginal_Gain_Calls",
        "Threshold_Checks", "Iterations"
    ])

    df_gain_breakdown = pd.DataFrame(gain_breakdown_data, columns=[
        "Algorithm", "Budget", "Total_g_Calls", "g({e})_Calls", "g(e|S)_Computations", "g(S∪{e})_Calls", "g(S)_Calls"
    ])

    df_memory_breakdown = pd.DataFrame(memory_breakdown_data, columns=[
        "Algorithm", "Budget", "Sizeof_Subregion_KB", "Base_Components", "Dual_Components", "IOT_Components",
        "Base_Memory_KB", "Dual_Memory_KB", "Total_Memory_KB"
    ])

    # Save results
    suffix = "_submodular" if args.use_submodular else "_simple"
    results_file = f"experiment_results{suffix}.csv"
    operations_file = f"operation_analysis{suffix}.csv"
    gain_file = f"gain_breakdown{suffix}.csv"
    memory_file = f"memory_breakdown{suffix}.csv"

    df_results.to_csv(results_file, index=False)
    df_operations.to_csv(operations_file, index=False)
    df_gain_breakdown.to_csv(gain_file, index=False)
    df_memory_breakdown.to_csv(memory_file, index=False)

    print(f"\nSaved results to:")
    print(f"  - {results_file}")
    print(f"  - {operations_file}")
    print(f"  - {gain_file}")
    print(f"  - {memory_file}")

    # Display results
    print(f"\n=== FINAL RESULTS ===")
    print(df_results)

    print(f"\n=== MEMORY BREAKDOWN ANALYSIS ===")
    print(df_memory_breakdown)
    print()
    print("Memory Formula Verification:")
    print("Greedy: M = #images × m × sizeof(subregion) + |S| × sizeof(subregion)")
    print("OT:     M = m × sizeof(subregion) + (|S| + |S'| + |I*|) × sizeof(subregion)")
    print(
        "IOT:    M = m × sizeof(subregion) + (|S| + |S'| + |I*|) × sizeof(subregion) + (max|S_τ| + max|S'_τ| + |S_b|) × sizeof(subregion)")

    for _, row in df_memory_breakdown.iterrows():
        alg = row['Algorithm']
        sizeof_sub = row['Sizeof_Subregion_KB']
        base_comp = row['Base_Components']
        dual_comp = row['Dual_Components']
        iot_comp = row['IOT_Components'] if not pd.isna(row['IOT_Components']) else 0
        total_mem = row['Total_Memory_KB']

        if alg == "Greedy":
            expected = base_comp * sizeof_sub
            print(f"{alg:8s}: {base_comp} × {sizeof_sub:.1f} = {expected:.1f} KB (actual: {total_mem:.1f})")
        elif alg == "OT":
            expected = base_comp * sizeof_sub + dual_comp * sizeof_sub
            print(
                f"{alg:8s}: {base_comp} × {sizeof_sub:.1f} + {dual_comp} × {sizeof_sub:.1f} = {expected:.1f} KB (actual: {total_mem:.1f})")
        elif alg == "IOT":
            expected = base_comp * sizeof_sub + dual_comp * sizeof_sub + iot_comp * sizeof_sub
            print(
                f"{alg:8s}: {base_comp} × {sizeof_sub:.1f} + {dual_comp} × {sizeof_sub:.1f} + {iot_comp} × {sizeof_sub:.1f} = {expected:.1f} KB (actual: {total_mem:.1f})")