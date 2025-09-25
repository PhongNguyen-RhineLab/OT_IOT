"""
memory_calculator.py - Theoretical Memory Analysis for Paper Algorithms
Paper: "Online approximate algorithms for Object detection under Budget allocation"
"""

import numpy as np


class MemoryCalculator:
    """
    Calculate theoretical memory usage according to paper formulas:

    Greedy: M = #images × m × sizeof(subregion) + |S| × sizeof(subregion)
    OT: M = m × sizeof(subregion) + |S+S'+I*| × sizeof(subregion)
    IOT: M = m × sizeof(subregion) + |S+S'+I*| × sizeof(subregion) + |candidates| × sizeof(subregion)
    """

    def __init__(self):
        # Estimate subregion size based on typical data structure
        self.estimated_subregion_size_kb = self._estimate_subregion_size()

    def _estimate_subregion_size(self):
        """
        Estimate sizeof(subregion) in KB based on data structure components:
        - mask: 224×224×4 bytes (float32) = ~200KB
        - saliency: 224×224×4 bytes = ~200KB
        - image reference: minimal (pointer/reference)
        - metadata: strings, floats (~1KB)

        Total: ~400KB per subregion (conservative estimate)
        """
        mask_size_kb = (224 * 224 * 4) / 1024  # ~200KB
        saliency_size_kb = (224 * 224 * 4) / 1024  # ~200KB
        metadata_kb = 1  # ~1KB

        total_kb = mask_size_kb + saliency_size_kb + metadata_kb
        return total_kb

    def calculate_greedy_memory(self, num_images, m_per_image, solution_size):
        """
        Greedy memory formula:
        M = #images × m × sizeof(subregion) + |S| × sizeof(subregion)

        Components:
        - All input subregions must be stored for comparison
        - Current solution set
        """
        input_components = num_images * m_per_image
        solution_components = solution_size

        memory_kb = (input_components + solution_components) * self.estimated_subregion_size_kb

        breakdown = {
            'input_subregions': input_components,
            'solution_set': solution_components,
            'total_components': input_components + solution_components,
            'memory_kb': memory_kb
        }

        return memory_kb, breakdown

    def calculate_ot_memory(self, m_per_image, max_S_size, max_S_prime_size, has_I_star=True):
        """
        OT memory formula:
        M = m × sizeof(subregion) + |S+S'+I*| × sizeof(subregion)

        Components:
        - Current batch of m subregions being processed
        - Dual candidate sets S and S'
        - Best singleton I*
        """
        batch_components = m_per_image
        dual_components = max_S_size + max_S_prime_size
        singleton_components = 1 if has_I_star else 0

        total_components = batch_components + dual_components + singleton_components
        memory_kb = total_components * self.estimated_subregion_size_kb

        breakdown = {
            'batch_subregions': batch_components,
            'candidate_S': max_S_size,
            'candidate_S_prime': max_S_prime_size,
            'best_singleton': singleton_components,
            'total_components': total_components,
            'memory_kb': memory_kb
        }

        return memory_kb, breakdown

    def calculate_iot_memory(self, m_per_image, max_S_size, max_S_prime_size,
                             has_I_star, max_threshold_candidates, baseline_solution_size):
        """
        IOT memory formula:
        M = m × sizeof(subregion) + |S+S'+I*| × sizeof(subregion) + |threshold_candidates| × sizeof(subregion)

        Components:
        - Base OT memory (m + S + S' + I*)
        - Additional threshold candidates from multiple τ values
        - Baseline solution from first pass
        """
        # Base OT components
        batch_components = m_per_image
        dual_components = max_S_size + max_S_prime_size
        singleton_components = 1 if has_I_star else 0

        # IOT-specific components
        threshold_candidates = max_threshold_candidates
        baseline_components = baseline_solution_size

        # Total memory calculation
        base_memory = (batch_components + dual_components + singleton_components) * self.estimated_subregion_size_kb
        iot_memory = (threshold_candidates + baseline_components) * self.estimated_subregion_size_kb
        total_memory = base_memory + iot_memory

        breakdown = {
            'batch_subregions': batch_components,
            'candidate_S': max_S_size,
            'candidate_S_prime': max_S_prime_size,
            'best_singleton': singleton_components,
            'threshold_candidates': threshold_candidates,
            'baseline_solution': baseline_components,
            'base_memory_kb': base_memory,
            'iot_memory_kb': iot_memory,
            'total_components': batch_components + dual_components + singleton_components + threshold_candidates + baseline_components,
            'memory_kb': total_memory
        }

        return total_memory, breakdown

    def calculate_memory_for_algorithm(self, algorithm, **kwargs):
        """
        Unified interface for memory calculation

        Args:
            algorithm: "Greedy", "OT", or "IOT"
            **kwargs: algorithm-specific parameters
        """
        if algorithm == "Greedy":
            return self.calculate_greedy_memory(
                kwargs.get('num_images', 0),
                kwargs.get('m_per_image', 0),
                kwargs.get('solution_size', 0)
            )
        elif algorithm == "OT":
            return self.calculate_ot_memory(
                kwargs.get('m_per_image', 0),
                kwargs.get('max_S_size', 0),
                kwargs.get('max_S_prime_size', 0),
                kwargs.get('has_I_star', False)
            )
        elif algorithm == "IOT":
            return self.calculate_iot_memory(
                kwargs.get('m_per_image', 0),
                kwargs.get('max_S_size', 0),
                kwargs.get('max_S_prime_size', 0),
                kwargs.get('has_I_star', False),
                kwargs.get('max_threshold_candidates', 0),
                kwargs.get('baseline_solution_size', 0)
            )
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

    def print_memory_analysis(self, algorithm, memory_kb, breakdown):
        """Print detailed memory analysis"""
        print(f"\n{'=' * 50}")
        print(f"{algorithm.upper()} MEMORY ANALYSIS")
        print('=' * 50)

        print(f"Estimated subregion size: {self.estimated_subregion_size_kb:.1f} KB")
        print(f"Total memory usage: {memory_kb:.1f} KB ({memory_kb / 1024:.1f} MB)")

        print(f"\nMemory Breakdown:")
        if algorithm == "Greedy":
            print(f"  Input subregions: {breakdown['input_subregions']} × {self.estimated_subregion_size_kb:.1f} KB")
            print(f"  Solution set: {breakdown['solution_set']} × {self.estimated_subregion_size_kb:.1f} KB")
            print(
                f"  Formula: ({breakdown['input_subregions']} + {breakdown['solution_set']}) × {self.estimated_subregion_size_kb:.1f}")

        elif algorithm == "OT":
            print(f"  Batch subregions: {breakdown['batch_subregions']} × {self.estimated_subregion_size_kb:.1f} KB")
            print(f"  Candidate S: {breakdown['candidate_S']} × {self.estimated_subregion_size_kb:.1f} KB")
            print(f"  Candidate S': {breakdown['candidate_S_prime']} × {self.estimated_subregion_size_kb:.1f} KB")
            print(f"  Best singleton: {breakdown['best_singleton']} × {self.estimated_subregion_size_kb:.1f} KB")
            print(
                f"  Formula: ({breakdown['batch_subregions']} + {breakdown['candidate_S']} + {breakdown['candidate_S_prime']} + {breakdown['best_singleton']}) × {self.estimated_subregion_size_kb:.1f}")

        elif algorithm == "IOT":
            print(f"  Base OT memory: {breakdown['base_memory_kb']:.1f} KB")
            print(f"    ├── Batch: {breakdown['batch_subregions']} × {self.estimated_subregion_size_kb:.1f} KB")
            print(f"    ├── Candidate S: {breakdown['candidate_S']} × {self.estimated_subregion_size_kb:.1f} KB")
            print(f"    ├── Candidate S': {breakdown['candidate_S_prime']} × {self.estimated_subregion_size_kb:.1f} KB")
            print(f"    └── Best singleton: {breakdown['best_singleton']} × {self.estimated_subregion_size_kb:.1f} KB")
            print(f"  IOT-specific memory: {breakdown['iot_memory_kb']:.1f} KB")
            print(
                f"    ├── Threshold candidates: {breakdown['threshold_candidates']} × {self.estimated_subregion_size_kb:.1f} KB")
            print(
                f"    └── Baseline solution: {breakdown['baseline_solution']} × {self.estimated_subregion_size_kb:.1f} KB")

    def compare_memory_usage(self, results_list):
        """
        Compare memory usage across different algorithms/configurations

        Args:
            results_list: List of (algorithm, memory_kb, breakdown) tuples
        """
        print(f"\n{'=' * 60}")
        print("MEMORY USAGE COMPARISON")
        print('=' * 60)

        print(f"{'Algorithm':<15} {'Memory (KB)':<12} {'Memory (MB)':<12} {'Components':<12}")
        print("-" * 60)

        for algorithm, memory_kb, breakdown in results_list:
            memory_mb = memory_kb / 1024
            components = breakdown['total_components']
            print(f"{algorithm:<15} {memory_kb:<12.1f} {memory_mb:<12.1f} {components:<12}")

        # Find memory ratios
        if len(results_list) > 1:
            base_memory = results_list[0][1]  # Use first as baseline
            print(f"\nMemory Ratios (vs {results_list[0][0]}):")
            for i, (algorithm, memory_kb, breakdown) in enumerate(results_list[1:], 1):
                ratio = memory_kb / base_memory
                print(f"  {algorithm}: {ratio:.2f}x")


def test_memory_calculator():
    """Test the memory calculator with example scenarios"""
    print("=== TESTING MEMORY CALCULATOR ===")

    calc = MemoryCalculator()
    print(f"Estimated subregion size: {calc.estimated_subregion_size_kb:.1f} KB")

    # Test scenarios
    scenarios = [
        # (algorithm, params)
        ("Greedy", {'num_images': 20, 'm_per_image': 8, 'solution_size': 15}),
        ("OT", {'m_per_image': 8, 'max_S_size': 5, 'max_S_prime_size': 3, 'has_I_star': True}),
        ("IOT", {'m_per_image': 8, 'max_S_size': 5, 'max_S_prime_size': 3, 'has_I_star': True,
                 'max_threshold_candidates': 12, 'baseline_solution_size': 8})
    ]

    results = []
    for algorithm, params in scenarios:
        memory_kb, breakdown = calc.calculate_memory_for_algorithm(algorithm, **params)
        calc.print_memory_analysis(algorithm, memory_kb, breakdown)
        results.append((algorithm, memory_kb, breakdown))

    # Compare all algorithms
    calc.compare_memory_usage(results)


if __name__ == "__main__":
    test_memory_calculator()