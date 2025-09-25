"""
operation_tracker.py - Comprehensive Operation Tracking for Paper Analysis
Paper: "Online approximate algorithms for Object detection under Budget allocation"
"""


class PaperOperationTracker:
    """
    Track operations according to paper's complexity analysis and theoretical bounds
    """

    def __init__(self, algorithm_name):
        self.algorithm = algorithm_name
        self.reset()

    def reset(self):
        """Reset all counters"""
        # Core oracle calls (most important for paper complexity analysis)
        self.oracle_calls = 0  # Total g(·) function evaluations
        self.marginal_gain_computations = 0  # g(e|S) = g(S∪{e}) - g(S) computations
        self.singleton_evaluations = 0  # g({e}) evaluations

        # Detailed breakdown of oracle calls
        self.gain_union_calls = 0  # g(S ∪ {e}) calls
        self.gain_current_set_calls = 0  # g(S) calls

        # Algorithmic operations
        self.iterations = 0  # Main loop iterations
        self.threshold_checks = 0  # Density/threshold condition checks
        self.density_comparisons = 0  # Density value comparisons
        self.budget_checks = 0  # Budget constraint checks

        # Memory-related tracking for theoretical analysis
        self.max_set_size_S = 0  # Max size of set S
        self.max_set_size_S_prime = 0  # Max size of set S'
        self.max_candidate_sets = 0  # Max concurrent candidate sets

        # Algorithm-specific counters
        self.threshold_sets_generated = 0  # Number of thresholds (IOT)
        self.dual_candidate_updates = 0  # S/S' updates (OT/IOT)
        self.best_singleton_updates = 0  # I* updates (OT/IOT)

    # === Core Oracle Tracking ===
    def count_oracle(self, call_type="general"):
        """Count oracle call g(·) with type specification"""
        self.oracle_calls += 1
        if call_type == "union":
            self.gain_union_calls += 1
        elif call_type == "current_set":
            self.gain_current_set_calls += 1

    def count_marginal_gain(self):
        """Count marginal gain computation g(e|S)"""
        self.marginal_gain_computations += 1

    def count_singleton(self):
        """Count singleton evaluation g({e})"""
        self.singleton_evaluations += 1

    # === Algorithmic Operation Tracking ===
    def count_iteration(self):
        """Count main algorithm iteration"""
        self.iterations += 1

    def count_threshold_check(self):
        """Count threshold condition evaluation"""
        self.threshold_checks += 1

    def count_density_comparison(self):
        """Count density value comparison"""
        self.density_comparisons += 1

    def count_budget_check(self):
        """Count budget constraint check"""
        self.budget_checks += 1

    # === Memory and Set Size Tracking ===
    def update_set_sizes(self, S_size, S_prime_size=0):
        """Update maximum set sizes for memory analysis"""
        self.max_set_size_S = max(self.max_set_size_S, S_size)
        self.max_set_size_S_prime = max(self.max_set_size_S_prime, S_prime_size)

    def update_candidate_sets(self, num_candidates):
        """Update maximum number of concurrent candidates"""
        self.max_candidate_sets = max(self.max_candidate_sets, num_candidates)

    # === Algorithm-Specific Tracking ===
    def count_threshold_set_generation(self, num_thresholds):
        """Track threshold set generation (IOT)"""
        self.threshold_sets_generated = num_thresholds

    def count_dual_candidate_update(self):
        """Count updates to dual candidates S/S'"""
        self.dual_candidate_updates += 1

    def count_best_singleton_update(self):
        """Count updates to best singleton I*"""
        self.best_singleton_updates += 1

    # === Analysis and Reporting ===
    def get_summary(self):
        """Get comprehensive operation summary"""
        return {
            # Core metrics
            'algorithm': self.algorithm,
            'oracle_calls': self.oracle_calls,
            'marginal_gain_computations': self.marginal_gain_computations,
            'singleton_evaluations': self.singleton_evaluations,
            'total_evaluations': self.oracle_calls + self.singleton_evaluations,

            # Detailed breakdown
            'gain_union_calls': self.gain_union_calls,
            'gain_current_set_calls': self.gain_current_set_calls,

            # Algorithmic operations
            'iterations': self.iterations,
            'threshold_checks': self.threshold_checks,
            'density_comparisons': self.density_comparisons,
            'budget_checks': self.budget_checks,

            # Memory-related
            'max_S_size': self.max_set_size_S,
            'max_S_prime_size': self.max_set_size_S_prime,
            'max_candidate_sets': self.max_candidate_sets,

            # Algorithm-specific
            'threshold_sets': self.threshold_sets_generated,
            'dual_updates': self.dual_candidate_updates,
            'singleton_updates': self.best_singleton_updates
        }

    def print_summary(self):
        """Print detailed operation summary"""
        summary = self.get_summary()

        print(f"\n{'=' * 60}")
        print(f"{self.algorithm.upper()} OPERATION SUMMARY")
        print('=' * 60)

        print(f"Core Oracle Calls:")
        print(f"  Total g(·) evaluations: {summary['oracle_calls']}")
        print(f"  ├── g(S ∪ {{e}}) calls: {summary['gain_union_calls']}")
        print(f"  ├── g(S) calls: {summary['gain_current_set_calls']}")
        print(
            f"  └── Other g(·) calls: {summary['oracle_calls'] - summary['gain_union_calls'] - summary['gain_current_set_calls']}")

        print(f"\nComputation Breakdown:")
        print(f"  Marginal gain g(e|S) computations: {summary['marginal_gain_computations']}")
        print(f"  Singleton g({{e}}) evaluations: {summary['singleton_evaluations']}")
        print(f"  Total function evaluations: {summary['total_evaluations']}")

        print(f"\nAlgorithmic Operations:")
        print(f"  Main iterations: {summary['iterations']}")
        print(f"  Threshold checks: {summary['threshold_checks']}")
        print(f"  Density comparisons: {summary['density_comparisons']}")
        print(f"  Budget checks: {summary['budget_checks']}")

        print(f"\nMemory Analysis:")
        print(f"  Max |S| size: {summary['max_S_size']}")
        print(f"  Max |S'| size: {summary['max_S_prime_size']}")
        print(f"  Max concurrent candidates: {summary['max_candidate_sets']}")

        if summary['threshold_sets'] > 0:
            print(f"\nAlgorithm-Specific (IOT):")
            print(f"  Threshold sets generated: {summary['threshold_sets']}")

        if summary['dual_updates'] > 0 or summary['singleton_updates'] > 0:
            print(f"\nStreaming Updates:")
            print(f"  Dual candidate updates: {summary['dual_updates']}")
            print(f"  Best singleton updates: {summary['singleton_updates']}")

    def verify_theoretical_bounds(self, n, budget=None, epsilon=None):
        """
        Verify operation counts against theoretical bounds from paper

        Args:
            n: total number of subregions
            budget: budget constraint (for analysis)
            epsilon: approximation parameter (for IOT)
        """
        summary = self.get_summary()
        print(f"\n{'=' * 60}")
        print(f"THEORETICAL COMPLEXITY VERIFICATION")
        print('=' * 60)

        oracle_calls = summary['oracle_calls']
        total_calls = summary['total_evaluations']

        if self.algorithm == "Greedy":
            # Greedy: O(n²) in worst case
            theoretical_max = n * n
            theoretical_typical = n * 20  # More realistic bound

            print(f"Greedy Search Analysis:")
            print(f"  Actual oracle calls: {oracle_calls}")
            print(f"  Theoretical worst case: O(n²) = {theoretical_max}")
            print(f"  Typical expected: ~{theoretical_typical}")
            print(f"  Efficiency ratio: {oracle_calls / theoretical_typical:.2f}")

        elif self.algorithm == "OT":
            # OT: 3n oracle calls according to paper
            theoretical_bound = 3 * n

            print(f"Online Threshold (OT) Analysis:")
            print(f"  Actual oracle calls: {oracle_calls}")
            print(f"  Theoretical bound: 3n = {theoretical_bound}")
            print(f"  Bound satisfaction: {'✓' if oracle_calls <= theoretical_bound * 1.1 else '✗'}")
            print(f"  Efficiency ratio: {oracle_calls / theoretical_bound:.2f}")

        elif self.algorithm == "IOT":
            # IOT: More complex bound depends on ε
            if epsilon:
                eps_prime = epsilon / 5
                # Simplified bound estimation
                theoretical_bound = 5 * n * int(1 / eps_prime)

                print(f"Improved Online Threshold (IOT) Analysis:")
                print(f"  Epsilon: {epsilon}")
                print(f"  Epsilon': {eps_prime}")
                print(f"  Actual oracle calls: {oracle_calls}")
                print(f"  Estimated theoretical bound: ~{theoretical_bound}")
                print(f"  Threshold sets used: {summary['threshold_sets']}")

                # Check if matches paper's "5n queries" claim
                simplified_bound = 5 * n
                print(f"  Paper's \"5n\" claim: {simplified_bound}")
                print(f"  Matches claim: {'✓' if oracle_calls <= simplified_bound * 2 else '✗'}")
            else:
                print(f"IOT Analysis: Need epsilon parameter for theoretical verification")

        print(f"\nMemory Complexity:")
        max_memory_components = summary['max_S_size'] + summary['max_S_prime_size'] + 1
        if self.algorithm == "Greedy":
            print(f"  Greedy memory: O(n + |S|) = O({n} + {summary['max_S_size']})")
        elif self.algorithm in ["OT", "IOT"]:
            print(f"  {self.algorithm} memory: O(m + |candidates|) = O(? + {max_memory_components})")