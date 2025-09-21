import time
import tracemalloc
import numpy as np
import pandas as pd
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights

from greedy_search import Greedy_Search
from ot_algo import OT_algorithm
from iot_algo import IOT_algorithm
from image_division import image_division
from gradcam import gradcam   # file gradcam.py chứa hàm mình đã viết

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

# ----------------- Main ----------------- #
if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True)

    # Model
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.eval()

    # Lấy 10 ảnh + saliency maps
    images, saliency_maps = [], []
    for img, label in testloader:
        cam = gradcam(model, img)
        images.append(img[0].permute(1,2,0).numpy())
        saliency_maps.append(cam)
        if len(images) >= 10:
            break

    budgets = [2000, 4000, 8000]   # pixel budget
    epsilons = [0.1, 0.3]
    results = []

    for B in budgets:
        # Greedy
        (S_gs, g_gs), t_gs, mem_gs = benchmark(
            Greedy_Search, images, saliency_maps, N=4, m=5, budget=B,
            cost_fn=cost_fn, gain_fn=gain_fn
        )
        results.append(["Greedy", B, "-", len(S_gs), g_gs, t_gs, mem_gs])

        # OT
        (S_ot, g_ot), t_ot, mem_ot = benchmark(
            OT_algorithm, images, saliency_maps, N=4, m=5, budget=B,
            cost_fn=cost_fn, gain_fn=gain_fn
        )
        results.append(["OT", B, "-", len(S_ot), g_ot, t_ot, mem_ot])

        # IOT
        for eps in epsilons:
            (S_iot, g_iot), t_iot, mem_iot = benchmark(
                IOT_algorithm, images, saliency_maps, N=4, m=5,
                budget=B, eps=eps, cost_fn=cost_fn, gain_fn=gain_fn
            )
            results.append(["IOT", B, eps, len(S_iot), g_iot, t_iot, mem_iot])

    df = pd.DataFrame(results, columns=[
        "Algorithm", "Budget", "Epsilon", "SetSize", "Gain", "Time(s)", "Memory(KB)"
    ])
    df.to_csv("experiment_results.csv", index=False)
    print("✅ Saved results to experiment_results.csv")
    print(df)
