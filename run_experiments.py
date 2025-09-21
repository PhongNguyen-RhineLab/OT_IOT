python
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

def gain_fn(region):
    return (region["saliency"] * region["mask"]).sum()

# ----------------- Dataset loader ----------------- #
def load_dataset(name: str, weights, root: str, num_samples: int):
    base_tf = weights.transforms()
    # base_tf = Resize(232) -> CenterCrop(224) -> ToTensor() -> Normalize(...)
    # We standardize: Resize(224,224) for simplicity (keeps saliency alignment)
    common_norm = transforms.Normalize(mean=base_tf.transforms[-1].mean,
                                       std=base_tf.transforms[-1].std)

    def rgb_pad_pipeline(extra=None):
        ops = [transforms.Resize((224, 224))]
        if extra:
            ops.extend(extra)
        ops += [transforms.ToTensor(), common_norm]
        return transforms.Compose(ops)

    if name == "cifar10":
        ds = CIFAR10(root=root, train=False, download=True,
                     transform=rgb_pad_pipeline())
    elif name == "cifar100":
        ds = CIFAR100(root=root, train=False, download=True,
                      transform=rgb_pad_pipeline())
    elif name == "stl10":
        ds = STL10(root=root, split="test", download=True,
                   transform=rgb_pad_pipeline())
    elif name == "mnist":
        ds = MNIST(root=root, train=False, download=True,
                   transform=rgb_pad_pipeline([transforms.Grayscale(num_output_channels=3)]))
    elif name == "fashionmnist":
        ds = FashionMNIST(root=root, train=False, download=True,
                          transform=rgb_pad_pipeline([transforms.Grayscale(num_output_channels=3)]))
    elif name == "imagenet":
        # User must supply prepared folder (e.g. `imagenet/val`)
        if not root:
            raise ValueError("Provide --data-root pointing to ImageNet split directory.")
        ds = ImageFolder(root=root, transform=base_tf)
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
    args = parser.parse_args()

    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.eval()

    images, saliency_maps = load_dataset(
        args.dataset, weights, args.data_root, args.num_samples
    )

    results = []
    for B in args.budgets:
        (S_gs, g_gs), t_gs, mem_gs = benchmark(
            Greedy_Search, images, saliency_maps, N=4, m=5,
            budget=B, cost_fn=cost_fn, gain_fn=gain_fn
        )
        results.append(["Greedy", args.dataset, B, "-", len(S_gs), g_gs, t_gs, mem_gs])

        (S_ot, g_ot), t_ot, mem_ot = benchmark(
            OT_algorithm, images, saliency_maps, N=4, m=5,
            budget=B, cost_fn=cost_fn, gain_fn=gain_fn
        )
        results.append(["OT", args.dataset, B, "-", len(S_ot), g_ot, t_ot, mem_ot])

        for eps in args.epsilons:
            (S_iot, g_iot), t_iot, mem_iot = benchmark(
                IOT_algorithm, images, saliency_maps, N=4, m=5,
                budget=B, eps=eps, cost_fn=cost_fn, gain_fn=gain_fn
            )
            results.append(["IOT", args.dataset, B, eps, len(S_iot), g_iot, t_iot, mem_iot])

    df = pd.DataFrame(results, columns=[
        "Algorithm", "Dataset", "Budget", "Epsilon", "SetSize", "Gain", "Time(s)", "Memory(KB)"
    ])
    out_file = "experiment_results.csv"
    df.to_csv(out_file, index=False)
    print(f"Saved results to {out_file}")
    print(df)
