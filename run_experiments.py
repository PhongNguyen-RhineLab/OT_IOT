import argparse
import time
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.datasets import CIFAR10, CIFAR100, STL10, MNIST, FashionMNIST, ImageFolder

from tracked_algorithms import Greedy_Search_Tracked, OT_Algorithm_Tracked, IOT_Algorithm_Tracked
from gradcam import gradcam


# Enhanced benchmark with operation tracking
def benchmark_with_tracking(func, *args, **kwargs):
    """Benchmark with runtime and operation tracking"""
    start = time.time()
    result, aux_data = func(*args, **kwargs)
    runtime = time.time() - start

    # Extract operation summary
    operation_summary = aux_data.get('operation_summary', {})

    return result, runtime, operation_summary, aux_data


# Cost & Gain functions
def cost_fn(region_or_list):
    if isinstance(region_or_list, list):
        if not region_or_list:
            return 0
        return sum(r["mask"].sum() for r in region_or_list)
    return region_or_list["mask"].sum()


def simple_gain_fn(region_or_list):
    if isinstance(region_or_list, list):
        if not region_or_list:
            return 0
        return sum((r["saliency"] * r["mask"]).sum() for r in region_or_list)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cifar10",
                        choices=["cifar10", "cifar100", "stl10", "mnist", "fashionmnist"])
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--budgets", type=int, nargs="+", default=[2000])
    parser.add_argument("--epsilons", type=float, nargs="+", default=[0.3])
    parser.add_argument("--m", type=int, default=5)
    args = parser.parse_args()

    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.eval()

    images, saliency_maps = load_dataset(
        args.dataset, weights, args.data_root, args.num_samples
    )

    print(f"Loaded {len(images)} images from {args.dataset}")

    results = []
    operation_results = []

    for B in args.budgets:
        print(f"\n{'=' * 60}")
        print(f"BUDGET: {B}")
        print(f"{'=' * 60}")

        # Test Greedy Search
        print(f"\n--- GREEDY SEARCH (Algorithm GS) ---")
        (S_gs, g_gs), t_gs, ops_gs, mem_gs = benchmark_with_tracking(
            Greedy_Search_Tracked,
            images, saliency_maps, N=4, m=args.m,
            budget=B, cost_fn=cost_fn, gain_fn=simple_gain_fn
        )

        results.append(["Greedy", args.dataset, B, "-", len(S_gs), g_gs, t_gs])
        operation_results.append(["Greedy", B, ops_gs.get('total_oracle_calls', 0),
                                  ops_gs.get('gain_calls', 0), ops_gs.get('marginal_gain_calls', 0),
                                  ops_gs.get('threshold_checks', 0), ops_gs.get('iterations', 0)])

        print(f"Results: |S|={len(S_gs)}, gain={g_gs:.3f}, time={t_gs:.3f}s")

        # Test OT Algorithm
        print(f"\n--- OT ALGORITHM (Algorithm 2) ---")
        (S_ot, g_ot), t_ot, ops_ot, mem_ot = benchmark_with_tracking(
            OT_Algorithm_Tracked,
            images, saliency_maps, N=4, m=args.m,
            budget=B, cost_fn=cost_fn, gain_fn=simple_gain_fn
        )

        results.append(["OT", args.dataset, B, "-", len(S_ot), g_ot, t_ot])
        operation_results.append(["OT", B, ops_ot.get('total_oracle_calls', 0),
                                  ops_ot.get('gain_calls', 0), ops_ot.get('marginal_gain_calls', 0),
                                  ops_ot.get('threshold_checks', 0), ops_ot.get('iterations', 0)])

        print(f"Results: |S|={len(S_ot)}, gain={g_ot:.3f}, time={t_ot:.3f}s")

        # Test IOT Algorithm
        for eps in args.epsilons:
            print(f"\n--- IOT ALGORITHM (Algorithm 3, ε={eps}) ---")
            (S_iot, g_iot), t_iot, ops_iot, mem_iot = benchmark_with_tracking(
                IOT_Algorithm_Tracked,
                images, saliency_maps, N=4, m=args.m,
                budget=B, eps=eps, cost_fn=cost_fn, gain_fn=simple_gain_fn
            )

            results.append(["IOT", args.dataset, B, eps, len(S_iot), g_iot, t_iot])
            operation_results.append(["IOT", B, ops_iot.get('total_oracle_calls', 0),
                                      ops_iot.get('gain_calls', 0), ops_iot.get('marginal_gain_calls', 0),
                                      ops_iot.get('threshold_checks', 0), ops_iot.get('iterations', 0)])

            print(f"Results: |S|={len(S_iot)}, gain={g_iot:.3f}, time={t_iot:.3f}s")

    # Save main results
    df_results = pd.DataFrame(results, columns=[
        "Algorithm", "Dataset", "Budget", "Epsilon", "SetSize", "Gain", "Time(s)"
    ])

    # Save operation tracking results
    df_operations = pd.DataFrame(operation_results, columns=[
        "Algorithm", "Budget", "Total_Oracle_Calls", "Gain_Calls", "Marginal_Gain_Calls",
        "Threshold_Checks", "Iterations"
    ])

    print(f"\n{'=' * 60}")
    print("FINAL RESULTS")
    print(f"{'=' * 60}")
    print(df_results)

    print(f"\n{'=' * 60}")
    print("OPERATION ANALYSIS")
    print(f"{'=' * 60}")
    print(df_operations)

    # Save to files
    df_results.to_csv("algorithm_results.csv", index=False)
    df_operations.to_csv("operation_analysis.csv", index=False)
    print(f"\nSaved results to algorithm_results.csv and operation_analysis.csv")

    # Theoretical vs Actual Analysis
    print(f"\n{'=' * 60}")
    print("THEORETICAL VS ACTUAL COMPLEXITY")
    print(f"{'=' * 60}")

    n = len(images) * args.m  # Total regions

    for _, row in df_operations.iterrows():
        alg = row['Algorithm']
        oracle_calls = row['Total_Oracle_Calls']

        if alg == "Greedy":
            theoretical = f"O(n^2) ≈ {n ** 2}"
        elif alg == "OT":
            theoretical = f"O(n) ≈ {3 * n}"  # 3n as per paper
        elif alg == "IOT":
            theoretical = f"O(5n) ≈ {5 * n}"  # 5n as per paper
        else:
            theoretical = "Unknown"

        print(f"{alg:8s}: Actual={oracle_calls:4d}, Theoretical={theoretical}")