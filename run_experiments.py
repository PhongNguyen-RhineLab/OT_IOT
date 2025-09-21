import time, tracemalloc
import numpy as np

# import các hàm thuật toán
from greedy_search import greedy_selection
from ot_algo import OT_algorithm
from iot_algo import IOT_algorithm


# ------------------ Sinh dữ liệu ------------------ #
def generate_data(num_regions=200):
    regions = [f"I{i}" for i in range(num_regions)]
    costs = {r: np.random.randint(1, 21) for r in regions}      # cost 1-20
    gains = {r: np.random.randint(10, 101) for r in regions}    # gain 10-100
    return regions, costs, gains


# ------------------ Benchmark ------------------ #
def benchmark(func, *args, **kwargs):
    tracemalloc.start()
    start_time = time.time()

    result = func(*args, **kwargs)

    runtime = time.time() - start_time
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return result, runtime, peak / 1024  # KB


# ------------------ Thực nghiệm ------------------ #
if __name__ == "__main__":
    regions, costs, gains = generate_data(200)

    budgets = [50, 100, 500, 1000]
    epsilons = [0.1, 0.5]

    for B in budgets:
        print(f"\n=== Budget = {B} ===")

        # GS
        S_gs, g_gs, *_ = benchmark(greedy_selection, regions, costs, gains, B)[0]
        _, t_gs, mem_gs = benchmark(greedy_selection, regions, costs, gains, B)
        print(f"GS -> |S|={len(S_gs)}, g(S)={g_gs}, "
              f"time={t_gs:.4f}s, memory={mem_gs:.2f} KB")

        # OT
        (S_ot, g_ot), t_ot, mem_ot = benchmark(OT_algorithm, regions, costs, gains, B)
        print(f"OT -> |S|={len(S_ot)}, g(S)={g_ot}, "
              f"time={t_ot:.4f}s, memory={mem_ot:.2f} KB")

        # IOT
        for eps in epsilons:
            (S_iot, g_iot), t_iot, mem_iot = benchmark(IOT_algorithm, regions, costs, gains, B, eps)
            print(f"IOT (eps={eps}) -> |S|={len(S_iot)}, g(S)={g_iot}, "
                  f"time={t_iot:.4f}s, memory={mem_iot:.2f} KB")
