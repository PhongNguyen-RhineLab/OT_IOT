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
from submodular_function import create_gain_function  # NEW IMPORT


# ----------------- Benchmark helper ----------------- #
def benchmark(func, *args, **kwargs):
    tracemalloc.start()
    start = time.time()
    result = func(*args, **kwargs)
    runtime = time.time() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, runtime, peak / 1024  # KB


# ----------------- Cost & Gain ----------------- #
def cost_fn(region):
    return region["mask"].sum()


# OLD simple gain function - keep as backup
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
        # Create feature extractor (use model without final classification layer)
        feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
        feature_extractor.eval()

        gain_fn = create_gain_function(
            model=model,
            feature_extractor=feature_extractor,
            original_images=images,
            lambda1=1.0,  # confidence weight
            lambda2=1.0,  # effectiveness weight
            lambda3=1.0,  # consistency weight
            lambda4=1.0  # collaboration weight
        )
    else:
        print("Using simple saliency-based gain function")
        gain_fn = simple_gain_fn

    results = []
    for B in args.budgets:
        print(f"\n=== Budget: {B} ===")

        (S_gs, g_gs), t_gs, mem_gs = benchmark(
            Greedy_Search, images, saliency_maps, N=4, m=5,
            budget=B, cost_fn=cost_fn, gain_fn=gain_fn
        )
        results.append(["Greedy", args.dataset, B, "-", len(S_gs), g_gs, t_gs, mem_gs])
        print(f"Greedy: |S|={len(S_gs)}, g(S)={g_gs:.3f}, time={t_gs:.3f}s")

        (S_ot, g_ot), t_ot, mem_ot = benchmark(
            OT_algorithm, images, saliency_maps, N=4, m=5,
            budget=B, cost_fn=cost_fn, gain_fn=gain_fn
        )
        results.append(["OT", args.dataset, B, "-", len(S_ot), g_ot, t_ot, mem_ot])
        print(f"OT: |S|={len(S_ot)}, g(S)={g_ot:.3f}, time={t_ot:.3f}s")

        for eps in args.epsilons:
            (S_iot, g_iot), t_iot, mem_iot = benchmark(
                IOT_algorithm, images, saliency_maps, N=4, m=5,
                budget=B, eps=eps, cost_fn=cost_fn, gain_fn=gain_fn
            )
            results.append(["IOT", args.dataset, B, eps, len(S_iot), g_iot, t_iot, mem_iot])
            print(f"IOT (ε={eps}): |S|={len(S_iot)}, g(S)={g_iot:.3f}, time={t_iot:.3f}s")

    df = pd.DataFrame(results, columns=[
        "Algorithm", "Dataset", "Budget", "Epsilon", "SetSize", "Gain", "Time(s)", "Memory(KB)"
    ])

    suffix = "_submodular" if args.use_submodular else "_simple"
    out_file = f"experiment_results{suffix}.csv"
    df.to_csv(out_file, index=False)
    print(f"\nSaved results to {out_file}")
    print(df)