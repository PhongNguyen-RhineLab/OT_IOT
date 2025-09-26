#!/usr/bin/env python3
"""
main_experiment.py - Complete Paper-Correct Experiment Runner
Paper: "Online approximate algorithms for Object detection under Budget allocation"

Usage:
    python main_experiment.py --dataset cifar10 --num-samples 20 --budgets 10 20 50
"""

import argparse
import time
import pandas as pd
import numpy as np
import sys
import os

# Import our modules
from dataset_loader import DatasetLoader
from paper_algorithms import paper_greedy_search, paper_ot_algorithm, paper_iot_algorithm
from cost_gain_functions import paper_cost_function, create_gain_function
from memory_calculator import MemoryCalculator
from operation_tracker import PaperOperationTracker


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Paper-Correct Implementation of Online Algorithms for Object Detection"
    )

    # Dataset parameters
    parser.add_argument("--dataset", default="cifar10",
                        choices=["cifar10", "cifar100", "stl10", "mnist", "fashionmnist"],
                        help="Dataset to use")
    parser.add_argument("--data-root", default="./data",
                        help="Root directory for dataset storage")
    parser.add_argument("--num-samples", type=int, default=20,
                        help="Number of images to process")

    # Algorithm parameters
    parser.add_argument("--budgets", type=int, nargs="+", default=[10, 20, 50],
                        help="Budget constraints to test")
    parser.add_argument("--epsilons", type=float, nargs="+", default=[0.1, 0.2],
                        help="Epsilon values for IOT algorithm")
    parser.add_argument("--m", type=int, default=8,
                        help="Number of subregions per image")
    parser.add_argument("--N", type=int, default=4,
                        help="Patch grid size (N×N patches)")

    # Timeout parameters
    parser.add_argument("--greedy-timeout", type=int, default=300,
                        help="Timeout for Greedy algorithm in seconds (default: 300s)")
    parser.add_argument("--ot-timeout", type=int, default=600,
                        help="Timeout for OT algorithm in seconds (default: 600s)")
    parser.add_argument("--iot-timeout", type=int, default=900,
                        help="Timeout for IOT algorithm in seconds (default: 900s)")

    # Experiment options
    parser.add_argument("--use-submodular", action="store_true",
                        help="Use full 4-component submodular function")
    parser.add_argument("--algorithms", nargs="+", default=["greedy", "ot", "iot"],
                        choices=["greedy", "ot", "iot"],
                        help="Algorithms to run")
    parser.add_argument("--output-prefix", default="paper_experiment",
                        help="Prefix for output files")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose output")

    return parser.parse_args()


def setup_experiment(args):
    """Setup experiment environment and data loading"""
    print("=" * 80)
    print("PAPER-CORRECT EXPERIMENT SETUP")
    print("=" * 80)
    print(f"Dataset: {args.dataset}")
    print(f"Samples: {args.num_samples}")
    print(f"Budgets: {args.budgets}")
    print(f"Algorithms: {args.algorithms}")
    print(f"Submodular function: {'Yes' if args.use_submodular else 'Simple saliency'}")

    # Load dataset
    print(f"\nLoading dataset...")
    loader = DatasetLoader()
    images, saliency_maps = loader.load_dataset(args.dataset, args.num_samples, args.data_root)

    # Setup gain function
    if args.use_submodular:
        print("Setting up full submodular gain function...")
        model_info = loader.get_model_info()
        gain_fn = create_gain_function(
            use_submodular=True,
            model=model_info['model'],
            feature_extractor=model_info['feature_extractor'],
            original_images=images,
            lambda_weights=[1.0, 1.0, 1.0, 1.0]
        )
    else:
        print("Using simple saliency-based gain function...")
        gain_fn = create_gain_function(use_submodular=False)

    return images, saliency_maps, gain_fn, loader.get_model_info()


def run_feasibility_check(images, saliency_maps, args, cost_fn, gain_fn):
    """Check if budgets are feasible with current cost scaling"""
    print(f"\nFEASIBILITY CHECK")
    print("-" * 30)

    # Test with small subset
    from image_division import image_division
    V_test = image_division(images[:2], saliency_maps[:2], args.N, args.m)

    if not V_test:
        print("ERROR: No subregions generated!")
        return False

    # Analyze costs
    test_costs = [cost_fn(r) for r in V_test[:20]]  # Sample 20 regions

    print(f"Cost analysis ({len(test_costs)} sample regions):")
    print(f"  Min cost: {min(test_costs):.6f}")
    print(f"  Max cost: {max(test_costs):.6f}")
    print(f"  Mean cost: {np.mean(test_costs):.6f}")
    print(f"  Median cost: {np.median(test_costs):.6f}")

    # Check budget feasibility
    all_feasible = True
    for budget in args.budgets:
        feasible_count = sum(1 for cost in test_costs if cost <= budget)
        feasibility_pct = feasible_count / len(test_costs) * 100
        print(f"  Budget {budget:3d}: {feasible_count:2d}/{len(test_costs)} regions feasible ({feasibility_pct:.0f}%)")

        if feasibility_pct < 10:  # Less than 10% feasible
            print(f"    ⚠️  WARNING: Very few regions feasible with budget {budget}")
            all_feasible = False

    if not all_feasible:
        print(f"\n💡 SUGGESTIONS:")
        suggested_budget = int(np.percentile(test_costs, 75))
        print(f"   Try budgets around: {[suggested_budget // 2, suggested_budget, suggested_budget * 2]}")

    return all_feasible


def run_single_experiment(algorithm_name, images, saliency_maps, args, budget,
                          cost_fn, gain_fn, epsilon=None):
    """Run single algorithm experiment with timeout support"""
    print(f"\nRunning {algorithm_name.upper()}" + (f" (ε={epsilon})" if epsilon else ""))
    print("-" * 50)

    try:
        if algorithm_name == "greedy":
            result = paper_greedy_search(
                images, saliency_maps, args.N, args.m, budget, cost_fn, gain_fn,
                timeout_seconds=args.greedy_timeout
            )
        elif algorithm_name == "ot":
            result = paper_ot_algorithm(
                images, saliency_maps, args.N, args.m, budget, cost_fn, gain_fn
            )
        elif algorithm_name == "iot":
            if epsilon is None:
                raise ValueError("Epsilon required for IOT algorithm")
            result = paper_iot_algorithm(
                images, saliency_maps, args.N, args.m, budget, epsilon, cost_fn, gain_fn
            )
        else:
            raise ValueError(f"Unknown algorithm: {algorithm_name}")

        # Add experiment metadata
        result.update({
            'dataset': args.dataset,
            'num_samples': args.num_samples,
            'budget': budget,
            'epsilon': epsilon,
            'N': args.N,
            'm': args.m,
            'use_submodular': args.use_submodular
        })

        # Add timeout info if available
        if 'completion_reason' in result:
            print(f"  Completion: {result['completion_reason']}")

        return result

    except Exception as e:
        print(f"ERROR in {algorithm_name}: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return None


def run_experiments(args):
    """Run complete experiment suite"""
    # Setup
    images, saliency_maps, gain_fn, model_info = setup_experiment(args)
    cost_fn = paper_cost_function

    # Feasibility check
    if not run_feasibility_check(images, saliency_maps, args, cost_fn, gain_fn):
        print("\n⚠️  Some budgets may not be feasible, but continuing experiment...")

    # Initialize results storage
    results = []
    detailed_operations = []
    memory_analysis = []

    # Run experiments
    print(f"\n" + "=" * 80)
    print("RUNNING EXPERIMENTS")
    print("=" * 80)

    for budget in args.budgets:
        print(f"\n{'=' * 20} BUDGET = {budget} {'=' * 20}")

        # Run each algorithm
        if "greedy" in args.algorithms:
            result = run_single_experiment(
                "greedy", images, saliency_maps, args, budget, cost_fn, gain_fn
            )
            if result:
                results.append(result)

        if "ot" in args.algorithms:
            result = run_single_experiment(
                "ot", images, saliency_maps, args, budget, cost_fn, gain_fn
            )
            if result:
                results.append(result)

        if "iot" in args.algorithms:
            for epsilon in args.epsilons:
                result = run_single_experiment(
                    "iot", images, saliency_maps, args, budget, cost_fn, gain_fn, epsilon
                )
                if result:
                    results.append(result)

    return results


def analyze_results(results, args):
    """Analyze and format experimental results"""
    if not results:
        print("No results to analyze!")
        return None, None, None

    print(f"\n" + "=" * 80)
    print("RESULTS ANALYSIS")
    print("=" * 80)

    # Create main results DataFrame
    main_results = []
    operation_details = []
    memory_details = []

    for result in results:
        # Main results
        algorithm = result['algorithm']
        if result.get('epsilon'):
            algorithm += f"(ε={result['epsilon']})"

        main_results.append([
            algorithm,
            result['dataset'],
            result['budget'],
            result.get('epsilon', '-'),
            len(result['solution']),
            result['gain'],
            result['runtime'],
            result['memory_kb']
        ])

        # Operation details
        ops = result['operations']
        operation_details.append([
            algorithm,
            result['budget'],
            ops['oracle_calls'],
            ops['marginal_gain_computations'],
            ops['singleton_evaluations'],
            ops['iterations'],
            ops['threshold_checks'],
            ops.get('num_thresholds', '-') if 'IOT' in algorithm else '-'
        ])

        # Memory details
        breakdown = result.get('memory_breakdown', {})
        memory_details.append([
            algorithm,
            result['budget'],
            breakdown.get('total_components', 0),
            result['memory_kb'],
            breakdown.get('input_subregions', breakdown.get('batch_subregions', 0)),
            breakdown.get('solution_set', breakdown.get('candidate_S', 0)),
            breakdown.get('threshold_candidates', 0) if 'IOT' in algorithm else 0
        ])

    # Create DataFrames
    df_results = pd.DataFrame(main_results, columns=[
        'Algorithm', 'Dataset', 'Budget', 'Epsilon', 'SolutionSize',
        'Gain', 'Runtime(s)', 'Memory(KB)'
    ])

    df_operations = pd.DataFrame(operation_details, columns=[
        'Algorithm', 'Budget', 'Oracle_Calls', 'Marginal_Gain_Comps',
        'Singleton_Evals', 'Iterations', 'Threshold_Checks', 'Num_Thresholds'
    ])

    df_memory = pd.DataFrame(memory_details, columns=[
        'Algorithm', 'Budget', 'Total_Components', 'Memory(KB)',
        'Input_Components', 'Solution_Components', 'Threshold_Components'
    ])

    # Display results
    print("\n1. PERFORMANCE COMPARISON:")
    print(df_results.to_string(index=False, float_format='%.3f'))

    print("\n2. OPERATION COMPLEXITY:")
    print(df_operations.to_string(index=False))

    print("\n3. MEMORY ANALYSIS:")
    print(df_memory.to_string(index=False, float_format='%.1f'))

    return df_results, df_operations, df_memory


def theoretical_verification(df_operations, args):
    """Verify results against theoretical bounds from paper"""
    print(f"\n4. THEORETICAL VERIFICATION:")
    print("-" * 50)

    n = args.num_samples * args.m  # Total subregions
    print(f"Total subregions (n): {n}")

    print(f"\nQuery Complexity Analysis:")
    for _, row in df_operations.iterrows():
        alg = row['Algorithm']
        oracle_calls = row['Oracle_Calls']

        if 'Greedy' in alg:
            theoretical_max = n * n  # O(n²) worst case
            typical_bound = n * 20  # More realistic
            efficiency = oracle_calls / typical_bound
            print(
                f"  {alg:<15}: {oracle_calls:6d} calls | O(n²)={theoretical_max} | Typical~{typical_bound} | Ratio={efficiency:.2f}")

        elif 'OT' == alg:
            theoretical_bound = 3 * n  # Paper claims 3n
            efficiency = oracle_calls / theoretical_bound
            status = "✓" if oracle_calls <= theoretical_bound * 1.2 else "✗"
            print(
                f"  {alg:<15}: {oracle_calls:6d} calls | Theory=3n={theoretical_bound} | Ratio={efficiency:.2f} {status}")

        elif 'IOT' in alg:
            # Extract epsilon from algorithm name
            eps = float(alg.split('ε=')[1].split(')')[0]) if 'ε=' in alg else 0.1
            eps_prime = eps / 5

            # Paper's "5n queries" claim
            simple_bound = 5 * n
            # More complex theoretical bound
            complex_bound = int(5 * n * (1 / eps_prime))

            efficiency_simple = oracle_calls / simple_bound
            efficiency_complex = oracle_calls / complex_bound

            print(f"  {alg:<15}: {oracle_calls:6d} calls | Paper=5n={simple_bound} | Complex~{complex_bound}")
            print(f"    └── Ratios: vs 5n={efficiency_simple:.2f}, vs complex={efficiency_complex:.2f}")

    print(f"\nApproximation Ratio Claims:")
    print(f"  Greedy: (1-1/e) ≈ 0.632 (optimal for submodular)")
    print(f"  OT: 1/8-ε ≈ 0.125 (paper Algorithm 2)")
    print(f"  IOT: 1/4-ε ≈ 0.25 OR 1/2-ε ≈ 0.5 (paper inconsistency)")


def save_results(df_results, df_operations, df_memory, args):
    """Save results to CSV files"""
    timestamp = int(time.time())
    suffix = "_submodular" if args.use_submodular else "_simple"

    files = {
        'results': f"{args.output_prefix}_results{suffix}_{timestamp}.csv",
        'operations': f"{args.output_prefix}_operations{suffix}_{timestamp}.csv",
        'memory': f"{args.output_prefix}_memory{suffix}_{timestamp}.csv"
    }

    df_results.to_csv(files['results'], index=False)
    df_operations.to_csv(files['operations'], index=False)
    df_memory.to_csv(files['memory'], index=False)

    print(f"\n" + "=" * 80)
    print("RESULTS SAVED TO:")
    for file_type, filename in files.items():
        print(f"  {file_type.capitalize()}: {filename}")
    print("=" * 80)

    return files


def main():
    """Main experiment execution"""
    args = parse_arguments()

    try:
        # Run experiments
        results = run_experiments(args)

        if not results:
            print("No results obtained. Check your parameters and try again.")
            sys.exit(1)

        # Analyze results
        df_results, df_operations, df_memory = analyze_results(results, args)

        # Theoretical verification
        theoretical_verification(df_operations, args)

        # Save results
        saved_files = save_results(df_results, df_operations, df_memory, args)

        # Final summary
        print(f"\n" + "=" * 80)
        print("EXPERIMENT COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print(f"Algorithms tested: {', '.join(args.algorithms)}")
        print(f"Budgets tested: {args.budgets}")
        print(f"Total experiments: {len(results)}")
        print(f"Dataset: {args.dataset} ({args.num_samples} samples)")
        print(f"Submodular function: {'Yes' if args.use_submodular else 'Simple saliency'}")

        # Performance highlights
        if len(df_results) > 0:
            best_gain = df_results.loc[df_results['Gain'].idxmax()]
            fastest = df_results.loc[df_results['Runtime(s)'].idxmin()]

            print(f"\nPerformance Highlights:")
            print(f"  Best gain: {best_gain['Algorithm']} with {best_gain['Gain']:.3f}")
            print(f"  Fastest: {fastest['Algorithm']} in {fastest['Runtime(s)']:.3f}s")

    except KeyboardInterrupt:
        print("\nExperiment interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nExperiment failed with error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()