"""
paper_algorithms.py - Paper-Correct Algorithm Implementations
Paper: "Online approximate algorithms for Object detection under Budget allocation"
"""

import time
from image_division import image_division
from operation_tracker import PaperOperationTracker
from memory_calculator import MemoryCalculator


def paper_greedy_search(images, saliency_maps, N, m, budget, cost_fn, gain_fn):
    """
    Algorithm GS (Greedy Search) from paper - Algorithm 4

    Returns dict with solution, gain, runtime, memory, and operation tracking
    """
    tracker = PaperOperationTracker("Greedy_GS")
    memory_calc = MemoryCalculator()

    print(f"Starting Algorithm GS (Greedy Search)")
    start_time = time.time()

    # Line 1: V ← ID(I, N, A, m)
    V = image_division(images, saliency_maps, N, m)
    n = len(V)
    print(f"  Generated {n} subregions")

    # Line 2: S ← ∅
    S = []

    # Line 3: U ← V
    U = list(V)

    iteration = 0
    # Line 4: repeat
    while U:
        tracker.count_iteration()
        iteration += 1

        if iteration % 100 == 0:
            print(f"    Iteration {iteration}: |U|={len(U)}, |S|={len(S)}")

        # Line 5: I_t^M ← arg max_{I^M ∈ U} g(I^M|S + I^M)/c(I^M)
        best_region = None
        best_density = -float('inf')

        for region in U:
            # Calculate marginal gain g(I^M|S)
            tracker.count_marginal_gain()
            if S:
                tracker.count_oracle("union")
                gain_with = gain_fn(S + [region])
                tracker.count_oracle("current_set")
                gain_without = gain_fn(S)
                marginal_gain = gain_with - gain_without
            else:
                tracker.count_singleton()
                marginal_gain = gain_fn([region])

            # Calculate density
            cost = cost_fn(region)
            if cost > 0:
                density = marginal_gain / cost
                tracker.count_density_comparison()
                if density > best_density:
                    best_density = density
                    best_region = region

        if best_region is None:
            print(f"    No valid region found at iteration {iteration}")
            break

        # Line 6: if c(S + I_t^M) ≤ B then
        current_cost = cost_fn(S)
        new_cost = cost_fn(best_region)
        tracker.count_budget_check()

        if current_cost + new_cost <= budget:
            # Line 7: S = S + I_t^M
            S.append(best_region)
            tracker.update_set_sizes(len(S))
        else:
            print(f"    Budget constraint violated at iteration {iteration}")
            break

        # Line 8: U ← U \ {I_t^M}
        U.remove(best_region)

        # Line 9: until U = ∅

    # Final evaluation
    if S:
        tracker.count_oracle()
        final_gain = gain_fn(S)
    else:
        final_gain = 0.0

    runtime = time.time() - start_time

    # Calculate memory usage
    memory_kb, memory_breakdown = memory_calc.calculate_greedy_memory(
        len(images), m, len(S)
    )

    print(f"  Algorithm GS completed in {runtime:.3f}s")
    print(f"  Solution: |S|={len(S)}, gain={final_gain:.3f}")
    print(f"  Operations: {tracker.oracle_calls} oracle calls, {tracker.iterations} iterations")

    return {
        'algorithm': 'Greedy_GS',
        'solution': S,
        'gain': final_gain,
        'runtime': runtime,
        'memory_kb': memory_kb,
        'memory_breakdown': memory_breakdown,
        'operations': tracker.get_summary()
    }


def paper_ot_algorithm(images, saliency_maps, N, m, budget, cost_fn, gain_fn):
    """
    Algorithm OT (Online Threshold) from paper - Algorithm 2

    Returns dict with solution, gain, runtime, memory, and operation tracking
    """
    tracker = PaperOperationTracker("OT")
    memory_calc = MemoryCalculator()

    print(f"Starting Algorithm OT (Online Threshold)")
    start_time = time.time()

    # Generate subregions (simulating stream)
    V = image_division(images, saliency_maps, N, m)
    n = len(V)
    print(f"  Processing {n} subregions in stream")

    # Line 1: V, S, S', I* ← ∅
    S = []
    S_prime = []
    I_star = None
    I_star_gain = 0.0

    # Lines 2-19: Process stream
    processed = 0
    for I_M in V:
        tracker.count_iteration()
        processed += 1

        if processed % 500 == 0:
            print(f"    Processed {processed}/{n} regions")

        # Line 9: I* = arg max_{I* ∈ {I*,I^M}} g(I*)
        tracker.count_singleton()
        singleton_gain = gain_fn([I_M])
        if I_star is None or singleton_gain > I_star_gain:
            I_star = I_M
            I_star_gain = singleton_gain
            tracker.count_best_singleton_update()

        # Line 6: S_d ← arg max_{S_d ∈ {S,S'}} g(I^M|S_d)
        if S:
            tracker.count_marginal_gain()
            tracker.count_oracle("union")
            gain_S_with = gain_fn(S + [I_M])
            tracker.count_oracle("current_set")
            gain_S_without = gain_fn(S)
            marginal_S = gain_S_with - gain_S_without
        else:
            marginal_S = singleton_gain

        if S_prime:
            tracker.count_marginal_gain()
            tracker.count_oracle("union")
            gain_Sp_with = gain_fn(S_prime + [I_M])
            tracker.count_oracle("current_set")
            gain_Sp_without = gain_fn(S_prime)
            marginal_Sp = gain_Sp_with - gain_Sp_without
        else:
            marginal_Sp = singleton_gain

        # Choose better candidate
        if marginal_S >= marginal_Sp:
            chosen_set = S
            marginal_chosen = marginal_S
            is_S = True
        else:
            chosen_set = S_prime
            marginal_chosen = marginal_Sp
            is_S = False

        # Line 7: if g(I^M|S_d)/c(I^M) ≥ g(S_d)/B then
        tracker.count_threshold_check()
        region_cost = cost_fn(I_M)

        if chosen_set:
            tracker.count_oracle("current_set")
            current_gain = gain_fn(chosen_set)
        else:
            current_gain = 0.0

        # Threshold condition
        if region_cost > 0 and marginal_chosen / region_cost >= current_gain / budget:
            # Line 8: S_d = S_d + I^M
            if is_S:
                S.append(I_M)
                tracker.update_set_sizes(len(S), len(S_prime))
            else:
                S_prime.append(I_M)
                tracker.update_set_sizes(len(S), len(S_prime))
            tracker.count_dual_candidate_update()

    # Lines 10-18: Final selection logic
    cost_S = cost_fn(S) if S else float('inf')
    cost_Sp = cost_fn(S_prime) if S_prime else float('inf')
    cost_Istar = cost_fn(I_star) if I_star else float('inf')

    # Evaluate feasible candidates
    candidates = []
    if cost_S <= budget:
        tracker.count_oracle()
        candidates.append((S, gain_fn(S)))
    if cost_Sp <= budget:
        tracker.count_oracle()
        candidates.append((S_prime, gain_fn(S_prime)))
    if cost_Istar <= budget:
        candidates.append(([I_star], I_star_gain))

    # Select best feasible solution
    if candidates:
        S_star, final_gain = max(candidates, key=lambda x: x[1])
    else:
        S_star, final_gain = [], 0.0

    runtime = time.time() - start_time

    # Calculate memory usage
    memory_kb, memory_breakdown = memory_calc.calculate_ot_memory(
        m, tracker.max_set_size_S, tracker.max_set_size_S_prime, I_star is not None
    )

    print(f"  Algorithm OT completed in {runtime:.3f}s")
    print(f"  Solution: |S*|={len(S_star)}, gain={final_gain:.3f}")
    print(f"  Operations: {tracker.oracle_calls} oracle calls")
    print(f"  Max sizes: |S|={tracker.max_set_size_S}, |S'|={tracker.max_set_size_S_prime}")

    return {
        'algorithm': 'OT',
        'solution': S_star,
        'gain': final_gain,
        'runtime': runtime,
        'memory_kb': memory_kb,
        'memory_breakdown': memory_breakdown,
        'operations': tracker.get_summary()
    }


def paper_iot_algorithm(images, saliency_maps, N, m, budget, epsilon, cost_fn, gain_fn):
    """
    Algorithm IOT (Improved Online Threshold) from paper - Algorithm 3

    Returns dict with solution, gain, runtime, memory, and operation tracking
    """
    tracker = PaperOperationTracker("IOT")
    memory_calc = MemoryCalculator()

    print(f"Starting Algorithm IOT (ε={epsilon})")
    start_time = time.time()

    # Line 2: ε' = ε/5
    eps_prime = epsilon / 5

    # Line 3: S_b, M ← OT(N, A, m, B) - First pass
    print("  Phase 1: Running OT for baseline...")
    ot_result = paper_ot_algorithm(images, saliency_maps, N, m, budget, cost_fn, gain_fn)
    S_b = ot_result['solution']
    M = max(ot_result['gain'], 1.0)  # Avoid division by zero

    # Add OT operations to IOT tracker
    ot_ops = ot_result['operations']
    tracker.oracle_calls += ot_ops['oracle_calls']
    tracker.marginal_gain_computations += ot_ops['marginal_gain_computations']
    tracker.singleton_evaluations += ot_ops['singleton_evaluations']

    print(f"    OT baseline: |S_b|={len(S_b)}, gain={M:.3f}")

    # Line 4: Generate threshold set T
    lower_bound = max(1e-6, (1 - eps_prime) * eps_prime * M / (2 * budget))
    upper_bound = min(M, 4 * M / (eps_prime * budget))

    # Generate thresholds: τ = (1-ε')^i as per paper formula
    T = []
    i = 0
    base_threshold = upper_bound

    while True:
        tau = (1 - eps_prime) ** i * base_threshold
        if tau < lower_bound or len(T) >= 15:  # Reasonable limit
            break
        T.append(tau)
        i += 1

    if not T:
        T = [lower_bound]

    tracker.count_threshold_set_generation(len(T))
    print(f"    Generated {len(T)} thresholds: [{T[0]:.2e}, ..., {T[-1]:.2e}]")

    # Lines 5-12: Second pass with multiple thresholds
    print("  Phase 2: Multi-threshold processing...")

    # Initialize candidate sets for each threshold
    candidates = {tau: {'S_tau': [], 'S_prime_tau': []} for tau in T}
    max_total_candidates = 0

    V = image_division(images, saliency_maps, N, m)

    processed = 0
    for I_M in V:
        tracker.count_iteration()
        processed += 1

        if processed % 1000 == 0:
            print(f"      Processed {processed}/{len(V)} regions")

        # Process each threshold
        for tau in T:
            S_tau = candidates[tau]['S_tau']
            S_prime_tau = candidates[tau]['S_prime_tau']

            # Line 10: X_τ ← arg max_{X_τ ∈ {S_τ,S'_τ}} g(I^M|X_τ)
            if S_tau:
                tracker.count_marginal_gain()
                tracker.count_oracle("union")
                gain_St_with = gain_fn(S_tau + [I_M])
                tracker.count_oracle("current_set")
                gain_St_without = gain_fn(S_tau)
                marginal_St = gain_St_with - gain_St_without
            else:
                tracker.count_singleton()
                marginal_St = gain_fn([I_M])

            if S_prime_tau:
                tracker.count_marginal_gain()
                tracker.count_oracle("union")
                gain_Spt_with = gain_fn(S_prime_tau + [I_M])
                tracker.count_oracle("current_set")
                gain_Spt_without = gain_fn(S_prime_tau)
                marginal_Spt = gain_Spt_with - gain_Spt_without
            else:
                marginal_Spt = marginal_St  # Same singleton

            # Choose better candidate
            if marginal_St >= marginal_Spt:
                X_tau = S_tau
                marginal_X = marginal_St
                target_key = 'S_tau'
            else:
                X_tau = S_prime_tau
                marginal_X = marginal_Spt
                target_key = 'S_prime_tau'

            # Line 11: if g(I^M|X_τ)/c(I^M) ≥ τ ∧ c(X_τ) ≤ B then
            tracker.count_threshold_check()
            tracker.count_budget_check()

            region_cost = cost_fn(I_M)
            current_cost = cost_fn(X_tau)

            if (region_cost > 0 and marginal_X / region_cost >= tau and
                    current_cost + region_cost <= budget):
                # Line 12: X_τ = X_τ + I^M
                candidates[tau][target_key] = X_tau + [I_M]

        # Track maximum candidates for memory analysis
        total_candidates = sum(len(candidates[tau]['S_tau']) + len(candidates[tau]['S_prime_tau']) for tau in T)
        max_total_candidates = max(max_total_candidates, total_candidates)
        tracker.update_candidate_sets(total_candidates)

    # Lines 13-15: Find best solution among all candidates
    print("  Phase 3: Final selection...")
    all_solutions = [S_b]  # Include baseline

    for tau in T:
        S_tau = candidates[tau]['S_tau']
        S_prime_tau = candidates[tau]['S_prime_tau']

        if cost_fn(S_tau) <= budget:
            all_solutions.append(S_tau)
        if cost_fn(S_prime_tau) <= budget:
            all_solutions.append(S_prime_tau)

    # Evaluate all feasible solutions
    if all_solutions:
        best_solution = None
        best_gain = -1.0

        for solution in all_solutions:
            if solution is not None:
                tracker.count_oracle()
                gain = gain_fn(solution) if solution else 0.0
                if gain > best_gain:
                    best_gain = gain
                    best_solution = solution

        S_star = best_solution if best_solution is not None else []
        final_gain = best_gain if best_gain >= 0 else 0.0
    else:
        S_star, final_gain = [], 0.0

    runtime = time.time() - start_time

    # Calculate memory usage
    memory_kb, memory_breakdown = memory_calc.calculate_iot_memory(
        m,
        ot_result['operations']['max_S_size'],
        ot_result['operations']['max_S_prime_size'],
        True,  # has_I_star
        max_total_candidates,
        len(S_b)
    )

    print(f"  Algorithm IOT completed in {runtime:.3f}s")
    print(f"  Solution: |S*|={len(S_star)}, gain={final_gain:.3f}")
    print(f"  Operations: {tracker.oracle_calls} total oracle calls")
    print(f"  Thresholds: {len(T)}, Max candidates: {max_total_candidates}")

    return {
        'algorithm': 'IOT',
        'solution': S_star,
        'gain': final_gain,
        'runtime': runtime,
        'memory_kb': memory_kb,
        'memory_breakdown': memory_breakdown,
        'operations': tracker.get_summary(),
        'epsilon': epsilon,
        'num_thresholds': len(T),
        'baseline_result': ot_result
    }


def test_algorithms():
    """Test all algorithms with dummy data"""
    print("=== TESTING PAPER ALGORITHMS ===")

    # Create dummy test data
    np.random.seed(42)  # Reproducible results
    test_images = [np.random.rand(224, 224, 3) for _ in range(3)]
    test_saliency = [np.random.rand(224, 224) for _ in range(3)]

    # Import cost/gain functions
    from cost_gain_functions import paper_cost_function, paper_gain_function_simple

    # Test parameters
    N, m = 4, 6
    budget = 12  # Should allow multiple regions
    epsilon = 0.2

    print(f"Test parameters: N={N}, m={m}, budget={budget}")

    # Test Greedy
    print(f"\n{'-' * 20} TESTING GREEDY {'-' * 20}")
    greedy_result = paper_greedy_search(
        test_images, test_saliency, N, m, budget,
        paper_cost_function, paper_gain_function_simple
    )

    # Test OT
    print(f"\n{'-' * 20} TESTING OT {'-' * 20}")
    ot_result = paper_ot_algorithm(
        test_images, test_saliency, N, m, budget,
        paper_cost_function, paper_gain_function_simple
    )

    # Test IOT
    print(f"\n{'-' * 20} TESTING IOT {'-' * 20}")
    iot_result = paper_iot_algorithm(
        test_images, test_saliency, N, m, budget, epsilon,
        paper_cost_function, paper_gain_function_simple
    )

    # Compare results
    print(f"\n{'=' * 50}")
    print("ALGORITHM COMPARISON")
    print('=' * 50)
    results = [greedy_result, ot_result, iot_result]

    print(f"{'Algorithm':<10} {'|S|':<5} {'Gain':<8} {'Time(s)':<8} {'Memory(KB)':<12} {'Oracles':<8}")
    print('-' * 60)
    for result in results:
        ops = result['operations']
        print(f"{result['algorithm']:<10} {len(result['solution']):<5} "
              f"{result['gain']:<8.1f} {result['runtime']:<8.3f} "
              f"{result['memory_kb']:<12.1f} {ops['oracle_calls']:<8}")

    # Verify theoretical bounds
    n = len(test_images) * m  # Total subregions
    print(f"\nTHEORETICAL VERIFICATION (n={n}):")

    for result in results:
        ops = result['operations']
        tracker = PaperOperationTracker(result['algorithm'])
        tracker.__dict__.update(ops)
        tracker.verify_theoretical_bounds(n, budget, iot_result.get('epsilon'))


if __name__ == "__main__":
    test_algorithms()