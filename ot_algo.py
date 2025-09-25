from image_division import image_division
from tracked_algorithms import OperationTracker


def OT_Algorithm_Corrected(images, saliency_maps, N, m, budget, cost_fn, gain_fn):
    tracker = OperationTracker("OT (Corrected)")

    V = image_division(images, saliency_maps, N, m)
    S, S_prime, I_star = [], [], None

    # Cache để tránh tính lại
    cached_gains = {}

    def get_cached_gain(region_list):
        key = tuple(sorted([r['id'] for r in region_list]))
        if key not in cached_gains:
            tracker.count_gain_call()
            cached_gains[key] = gain_fn(region_list)
        return cached_gains[key]

    for I_M in V:
        tracker.count_iteration()

        # Tính g(S∪{I^M}) - 1 oracle call
        gain_S_union = get_cached_gain(S + [I_M])

        # Tính g(S'∪{I^M}) - 1 oracle call
        gain_S_prime_union = get_cached_gain(S_prime + [I_M])

        # Lấy g(S) và g(S') từ cache (đã tính từ iteration trước)
        gain_S = get_cached_gain(S)
        gain_S_prime = get_cached_gain(S_prime)

        # Marginal gains - không cần oracle calls thêm
        marginal_S = gain_S_union - gain_S
        marginal_S_prime = gain_S_prime_union - gain_S_prime

        # Chọn candidate tốt hơn
        if marginal_S >= marginal_S_prime:
            S_d = S
            marginal_g = marginal_S
            current_gain = gain_S
            is_S = True
        else:
            S_d = S_prime
            marginal_g = marginal_S_prime
            current_gain = gain_S_prime
            is_S = False

        # Threshold check
        region_cost = cost_fn(I_M)
        if marginal_g / region_cost >= current_gain / budget:
            if is_S:
                S.append(I_M)
            else:
                S_prime.append(I_M)

        # Update I* - 1 oracle call cho singleton
        tracker.count_singleton_gain_call()
        singleton_gain = gain_fn([I_M])
        if I_star is None or singleton_gain > get_cached_gain([I_star]):
            I_star = I_M

    # Final selection
    final_candidates = []
    for candidate in [S, S_prime, [I_star] if I_star else []]:
        if candidate and sum(cost_fn(x) for x in candidate) <= budget:
            final_candidates.append((candidate, get_cached_gain(candidate)))

    return max(final_candidates, key=lambda x: x[1]) if final_candidates else ([], 0)

# Oracle calls cho OT:
# Mỗi iteration: 2 g(S∪{I^M}) calls + 1 g({I^M}) call = 3 calls
# Total: 3n calls (chính xác như paper!)


def OT_algorithm(images, saliency_maps, N, m, budget, cost_fn, gain_fn):
    """
    Balanced OT: Theory-correct implementation with reasonable optimizations.
    Follows Algorithm 2 exactly while minimizing computational overhead.
    """
    print("OT: Starting balanced (theory-correct + optimized) algorithm")
    V = image_division(images, saliency_maps, N, m)

    if not V:
        return ([], 0), {'S_size': 0, 'S_prime_size': 0, 'has_I_star': False}

    n = len(V)
    print(f"OT: Processing {n} regions following Algorithm 2")

    # Initialize as in Algorithm 2
    S, S_prime = [], []
    I_star = None
    I_star_gain = 0

    # OPTIMIZATION 1: Precompute singleton values to avoid recomputation
    singleton_data = {}
    infeasible_regions = set()

    for region in V:
        cost = cost_fn(region)
        if cost > budget:
            infeasible_regions.add(region['id'])
            continue

        gain = gain_fn([region])
        singleton_data[region['id']] = {
            'cost': cost,
            'gain': gain,
            'density': gain / max(cost, 1e-6)
        }

    print(f"OT: {len(singleton_data)} feasible regions out of {n}")

    # OPTIMIZATION 2: Cache for gain computations to avoid redundant calls
    gain_cache = {}

    def cached_gain(region_list):
        """Cached gain computation to minimize calls"""
        if not region_list:
            return 0
        key = tuple(sorted([r['id'] for r in region_list]))
        if key not in gain_cache:
            gain_cache[key] = gain_fn(region_list)
        return gain_cache[key]

    def exact_marginal_gain(element, current_set):
        """EXACT marginal gain as required by Algorithm 2: g(e|S) = g(S∪{e}) - g(S)"""
        if not current_set:
            return singleton_data[element['id']]['gain']

        gain_with = cached_gain(current_set + [element])
        gain_without = cached_gain(current_set)
        return gain_with - gain_without

    # Main loop following Algorithm 2 EXACTLY
    processed = 0
    for I_M in V:
        processed += 1
        if processed % max(1, n // 10) == 0:
            print(f"OT: Processed {processed}/{n} regions")

        # Skip infeasible regions early (OPTIMIZATION 3)
        if I_M['id'] in infeasible_regions:
            continue

        region_data = singleton_data[I_M['id']]

        # Line 9: Update I* (best singleton) - EXACT as in paper
        if I_star is None or region_data['gain'] > I_star_gain:
            I_star = I_M
            I_star_gain = region_data['gain']

        # Line 6: Choose S_d with higher marginal gain - EXACT as in Algorithm 2
        marginal_gain_S = exact_marginal_gain(I_M, S)
        marginal_gain_S_prime = exact_marginal_gain(I_M, S_prime)

        if marginal_gain_S >= marginal_gain_S_prime:
            S_d = S
            marginal_g = marginal_gain_S
            is_S = True
        else:
            S_d = S_prime
            marginal_g = marginal_gain_S_prime
            is_S = False

        # Line 7: EXACT threshold condition g(I^M|S_d)/c(I^M) ≥ g(S_d)/B
        current_gain = cached_gain(S_d)
        region_cost = region_data['cost']

        if marginal_g / region_cost >= current_gain / budget:
            # Line 8: Add to selected candidate
            if is_S:
                S = S + [I_M]
            else:
                S_prime = S_prime + [I_M]

    print("OT: Final selection following Lines 10-18 of Algorithm 2...")

    # Lines 10-18: Final selection EXACTLY as in Algorithm 2
    cost_S = sum(singleton_data[r['id']]['cost'] for r in S if r['id'] in singleton_data)
    cost_S_prime = sum(singleton_data[r['id']]['cost'] for r in S_prime if r['id'] in singleton_data)
    cost_I_star = singleton_data[I_star['id']]['cost'] if I_star and I_star['id'] in singleton_data else float('inf')

    # Compute final exact gains
    gain_S = cached_gain(S)
    gain_S_prime = cached_gain(S_prime)
    gain_I_star = I_star_gain if I_star else 0

    # Follow Algorithm 2 cases exactly
    if cost_S <= budget and cost_S_prime <= budget:
        # Line 11: Both feasible
        candidates = [(S, gain_S), (S_prime, gain_S_prime)]
        if cost_I_star <= budget:
            candidates.append(([I_star], gain_I_star))
        S_star, g_star = max(candidates, key=lambda x: x[1])

    elif cost_S > budget and cost_S_prime > budget:
        # Lines 13-15: Both infeasible
        S1 = get_feasible_prefix(S, budget, singleton_data)
        S2 = get_feasible_prefix(S_prime, budget, singleton_data)

        candidates = []
        if S1:
            candidates.append((S1, cached_gain(S1)))
        if S2:
            candidates.append((S2, cached_gain(S2)))
        if cost_I_star <= budget:
            candidates.append(([I_star], gain_I_star))

        S_star, g_star = max(candidates, key=lambda x: x[1]) if candidates else ([], 0)

    else:
        # Lines 16-18: One feasible, one infeasible
        if cost_S <= budget:
            feasible, feasible_gain = S, gain_S
            infeasible = S_prime
        else:
            feasible, feasible_gain = S_prime, gain_S_prime
            infeasible = S

        infeasible_prefix = get_feasible_prefix(infeasible, budget, singleton_data)

        candidates = [(feasible, feasible_gain)]
        if infeasible_prefix:
            candidates.append((infeasible_prefix, cached_gain(infeasible_prefix)))
        if cost_I_star <= budget:
            candidates.append(([I_star], gain_I_star))

        S_star, g_star = max(candidates, key=lambda x: x[1])

    print(f"OT: Theory-correct completed. Final gain = {g_star}, |S*| = {len(S_star)}")
    print(f"OT: Cache hits: {len(gain_cache)} unique gain computations")

    # Memory aux data
    memory_aux = {
        'S_size': len(S),
        'S_prime_size': len(S_prime),
        'has_I_star': I_star is not None
    }

    return (S_star, g_star), memory_aux


def get_feasible_prefix(S, budget, singleton_data):
    """Get largest prefix that satisfies budget"""
    prefix = []
    total_cost = 0

    for region in S:
        if region['id'] not in singleton_data:
            continue
        region_cost = singleton_data[region['id']]['cost']
        if total_cost + region_cost <= budget:
            prefix.append(region)
            total_cost += region_cost
        else:
            break

    return prefix