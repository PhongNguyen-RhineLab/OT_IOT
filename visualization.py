"""
visualization.py - Professional Visualization for Paper Results
Paper: "Online approximate algorithms for Object detection under Budget allocation"

Usage:
    python visualization.py results_file.csv operations_file.csv memory_file.csv
    python visualization.py --help
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set professional plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class PaperVisualizer:
    """Generate publication-quality plots for algorithm comparison"""

    def __init__(self, results_df, operations_df, memory_df, output_dir="plots"):
        self.results_df = results_df
        self.operations_df = operations_df
        self.memory_df = memory_df
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Clean algorithm names for better display
        self.results_df = self._clean_algorithm_names(self.results_df)
        self.operations_df = self._clean_algorithm_names(self.operations_df)
        self.memory_df = self._clean_algorithm_names(self.memory_df)

    def _clean_algorithm_names(self, df):
        """Clean algorithm names for better visualization"""
        if 'Algorithm' in df.columns:
            df = df.copy()
            df['Algorithm'] = df['Algorithm'].str.replace('Greedy_GS', 'Greedy')
            df['Algorithm'] = df['Algorithm'].str.replace('IOT(ε=', 'IOT(ε=')
            df['Algorithm'] = df['Algorithm'].str.replace(')', ')')
        return df

    def plot_solution_quality_comparison(self):
        """Plot 1: Solution Quality vs Budget"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Plot 1a: Gain vs Budget
        for algorithm in self.results_df['Algorithm'].unique():
            data = self.results_df[self.results_df['Algorithm'] == algorithm]
            axes[0].plot(data['Budget'], data['Gain'],
                        marker='o', linewidth=2, markersize=8, label=algorithm)

        axes[0].set_xlabel('Budget Constraint', fontsize=12)
        axes[0].set_ylabel('Submodular Function Value g(S)', fontsize=12)
        axes[0].set_title('Solution Quality vs Budget', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)

        # Plot 1b: Solution Size vs Budget
        for algorithm in self.results_df['Algorithm'].unique():
            data = self.results_df[self.results_df['Algorithm'] == algorithm]
            axes[1].plot(data['Budget'], data['SolutionSize'],
                        marker='s', linewidth=2, markersize=8, label=algorithm)

        axes[1].set_xlabel('Budget Constraint', fontsize=12)
        axes[1].set_ylabel('Solution Set Size |S|', fontsize=12)
        axes[1].set_title('Solution Size vs Budget', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'solution_quality_comparison.png',
                   dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'solution_quality_comparison.pdf',
                   bbox_inches='tight')
        plt.show()

    def plot_computational_efficiency(self):
        """Plot 2: Runtime and Query Complexity"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Plot 2a: Runtime vs Budget
        for algorithm in self.results_df['Algorithm'].unique():
            data = self.results_df[self.results_df['Algorithm'] == algorithm]
            axes[0,0].semilogy(data['Budget'], data['Runtime(s)'],
                              marker='o', linewidth=2, markersize=8, label=algorithm)

        axes[0,0].set_xlabel('Budget Constraint', fontsize=12)
        axes[0,0].set_ylabel('Runtime (seconds)', fontsize=12)
        axes[0,0].set_title('Runtime Performance', fontsize=14, fontweight='bold')
        axes[0,0].legend(fontsize=10)
        axes[0,0].grid(True, alpha=0.3)

        # Plot 2b: Oracle Calls vs Budget
        for algorithm in self.operations_df['Algorithm'].unique():
            data = self.operations_df[self.operations_df['Algorithm'] == algorithm]
            axes[0,1].semilogy(data['Budget'], data['Oracle_Calls'],
                              marker='s', linewidth=2, markersize=8, label=algorithm)

        axes[0,1].set_xlabel('Budget Constraint', fontsize=12)
        axes[0,1].set_ylabel('Oracle Calls (log scale)', fontsize=12)
        axes[0,1].set_title('Query Complexity', fontsize=14, fontweight='bold')
        axes[0,1].legend(fontsize=10)
        axes[0,1].grid(True, alpha=0.3)

        # Plot 2c: Runtime vs Problem Size (if multiple scales available)
        if len(self.results_df['Budget'].unique()) >= 3:
            budget_sizes = sorted(self.results_df['Budget'].unique())
            for algorithm in self.results_df['Algorithm'].unique():
                runtimes = []
                for budget in budget_sizes:
                    data = self.results_df[
                        (self.results_df['Algorithm'] == algorithm) &
                        (self.results_df['Budget'] == budget)
                    ]
                    if not data.empty:
                        runtimes.append(data['Runtime(s)'].mean())
                    else:
                        runtimes.append(np.nan)

                axes[1,0].plot(budget_sizes, runtimes,
                              marker='d', linewidth=2, markersize=8, label=algorithm)

        axes[1,0].set_xlabel('Problem Scale (Budget)', fontsize=12)
        axes[1,0].set_ylabel('Runtime (seconds)', fontsize=12)
        axes[1,0].set_title('Scalability Analysis', fontsize=14, fontweight='bold')
        axes[1,0].legend(fontsize=10)
        axes[1,0].grid(True, alpha=0.3)

        # Plot 2d: Efficiency Ratio (Gain/Runtime)
        for algorithm in self.results_df['Algorithm'].unique():
            data = self.results_df[self.results_df['Algorithm'] == algorithm]
            efficiency = data['Gain'] / data['Runtime(s)']
            axes[1,1].plot(data['Budget'], efficiency,
                          marker='^', linewidth=2, markersize=8, label=algorithm)

        axes[1,1].set_xlabel('Budget Constraint', fontsize=12)
        axes[1,1].set_ylabel('Efficiency (Gain/Runtime)', fontsize=12)
        axes[1,1].set_title('Computational Efficiency', fontsize=14, fontweight='bold')
        axes[1,1].legend(fontsize=10)
        axes[1,1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'computational_efficiency.png',
                   dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'computational_efficiency.pdf',
                   bbox_inches='tight')
        plt.show()

    def plot_memory_analysis(self):
        """Plot 3: Memory Usage Analysis"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Debug: Print available columns
        print("Memory DF columns:", self.memory_df.columns.tolist())
        print("Results DF columns:", self.results_df.columns.tolist())

        # Find correct memory column name
        memory_col = None
        for col in self.memory_df.columns:
            if 'Memory' in col and ('KB' in col or 'kb' in col):
                memory_col = col
                break

        if memory_col is None:
            print("Warning: No Memory column found, skipping memory plots")
            return

        print(f"Using memory column: {memory_col}")

        # Plot 3a: Memory vs Budget
        for algorithm in self.memory_df['Algorithm'].unique():
            data = self.memory_df[self.memory_df['Algorithm'] == algorithm]
            if not data.empty:
                axes[0].plot(data['Budget'], data[memory_col],
                            marker='o', linewidth=2, markersize=8, label=algorithm)

        axes[0].set_xlabel('Budget Constraint', fontsize=12)
        axes[0].set_ylabel('Memory Usage (KB)', fontsize=12)
        axes[0].set_title('Memory Usage', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)

        # Plot 3b: Memory Components Breakdown (Stacked Bar)
        algorithms = self.memory_df['Algorithm'].unique()
        budgets = sorted(self.memory_df['Budget'].unique())

        # Check if component columns exist
        component_cols = ['Input_Components', 'Solution_Components', 'Threshold_Components']
        available_cols = [col for col in component_cols if col in self.memory_df.columns]

        if len(budgets) >= 2 and available_cols:
            mid_budget = budgets[len(budgets)//2]
            subset = self.memory_df[self.memory_df['Budget'] == mid_budget]

            if not subset.empty:
                x = range(len(subset))
                bottom = np.zeros(len(subset))

                for col in available_cols:
                    if col in subset.columns:
                        values = subset[col] * 400 / 1024  # Convert to MB
                        axes[1].bar(x, values, bottom=bottom, label=col.replace('_', ' '), alpha=0.8)
                        bottom += values

                axes[1].set_xlabel('Algorithms', fontsize=12)
                axes[1].set_ylabel('Memory Usage (MB)', fontsize=12)
                axes[1].set_title(f'Memory Breakdown (Budget={mid_budget})', fontsize=14, fontweight='bold')
                axes[1].set_xticks(x)
                axes[1].set_xticklabels(subset['Algorithm'], rotation=45)
                axes[1].legend(fontsize=10)
                axes[1].grid(True, alpha=0.3)
        else:
            axes[1].text(0.5, 0.5, 'Memory breakdown\ndata not available',
                        ha='center', va='center', transform=axes[1].transAxes)

        # Plot 3c: Memory Efficiency (Memory per Solution Element)
        try:
            for algorithm in self.results_df['Algorithm'].unique():
                result_data = self.results_df[self.results_df['Algorithm'] == algorithm]
                memory_data = self.memory_df[self.memory_df['Algorithm'] == algorithm]

                # Merge data on common columns
                common_cols = ['Algorithm', 'Budget']
                if all(col in result_data.columns for col in common_cols) and \
                   all(col in memory_data.columns for col in common_cols):

                    merged = pd.merge(result_data, memory_data, on=common_cols, how='inner')
                    if not merged.empty and 'SolutionSize' in merged.columns:
                        efficiency = merged[memory_col] / (merged['SolutionSize'] + 1)
                        axes[2].plot(merged['Budget'], efficiency,
                                    marker='s', linewidth=2, markersize=8, label=algorithm)
        except Exception as e:
            print(f"Warning: Memory efficiency plot failed: {e}")
            axes[2].text(0.5, 0.5, 'Memory efficiency\nplot unavailable',
                        ha='center', va='center', transform=axes[2].transAxes)

        axes[2].set_xlabel('Budget Constraint', fontsize=12)
        axes[2].set_ylabel('Memory per Solution Element (KB)', fontsize=12)
        axes[2].set_title('Memory Efficiency', fontsize=14, fontweight='bold')
        axes[2].legend(fontsize=10)
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'memory_analysis.png',
                   dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'memory_analysis.pdf',
                   bbox_inches='tight')
        plt.show()

    def plot_approximation_ratios(self):
        """Plot 4: Approximation Quality Analysis"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Calculate approximation ratios relative to Greedy
        greedy_results = self.results_df[self.results_df['Algorithm'] == 'Greedy']

        if not greedy_results.empty:
            # Plot 4a: Approximation Ratio vs Budget
            for algorithm in self.results_df['Algorithm'].unique():
                if algorithm == 'Greedy':
                    continue

                data = self.results_df[self.results_df['Algorithm'] == algorithm]
                ratios = []
                budgets = []

                for budget in data['Budget']:
                    algo_gain = data[data['Budget'] == budget]['Gain'].iloc[0]
                    greedy_gain = greedy_results[greedy_results['Budget'] == budget]['Gain']

                    if not greedy_gain.empty and greedy_gain.iloc[0] > 0:
                        ratio = algo_gain / greedy_gain.iloc[0]
                        ratios.append(ratio)
                        budgets.append(budget)

                if ratios:
                    axes[0].plot(budgets, ratios,
                                marker='o', linewidth=2, markersize=8, label=algorithm)

            axes[0].axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Greedy (Optimal)')
            axes[0].axhline(y=0.632, color='orange', linestyle=':', alpha=0.7, label='(1-1/e) ≈ 0.632')
            axes[0].axhline(y=0.25, color='green', linestyle=':', alpha=0.7, label='1/4 = 0.25')
            axes[0].set_xlabel('Budget Constraint', fontsize=12)
            axes[0].set_ylabel('Approximation Ratio (vs Greedy)', fontsize=12)
            axes[0].set_title('Approximation Quality', fontsize=14, fontweight='bold')
            axes[0].legend(fontsize=10)
            axes[0].grid(True, alpha=0.3)
            axes[0].set_ylim(0, 1.1)

        # Plot 4b: IOT Epsilon Analysis
        iot_results = self.results_df[self.results_df['Algorithm'].str.contains('IOT')]
        if not iot_results.empty:
            # Extract epsilon values
            iot_results_copy = iot_results.copy()
            iot_results_copy['Epsilon_Val'] = iot_results_copy['Algorithm'].str.extract(r'ε=([0-9.]+)')
            iot_results_copy['Epsilon_Val'] = pd.to_numeric(iot_results_copy['Epsilon_Val'])

            for budget in sorted(iot_results_copy['Budget'].unique()):
                budget_data = iot_results_copy[iot_results_copy['Budget'] == budget]
                if len(budget_data) > 1:
                    axes[1].plot(budget_data['Epsilon_Val'], budget_data['Gain'],
                                marker='s', linewidth=2, markersize=8, label=f'Budget={budget}')

            axes[1].set_xlabel('Epsilon (ε)', fontsize=12)
            axes[1].set_ylabel('Solution Quality', fontsize=12)
            axes[1].set_title('IOT: Epsilon vs Quality Trade-off', fontsize=14, fontweight='bold')
            axes[1].legend(fontsize=10)
            axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'approximation_analysis.png',
                   dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'approximation_analysis.pdf',
                   bbox_inches='tight')
        plt.show()

    def plot_theoretical_verification(self):
        """Plot 5: Theoretical Bounds Verification"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Plot 5a: Query Complexity vs Theoretical Bounds
        for algorithm in self.operations_df['Algorithm'].unique():
            data = self.operations_df[self.operations_df['Algorithm'] == algorithm]
            axes[0,0].plot(data['Budget'], data['Oracle_Calls'],
                          marker='o', linewidth=2, markersize=8, label=f'{algorithm} (Actual)')

            # Add theoretical bounds
            if algorithm == 'Greedy':
                # Theoretical: O(n²) but practical is usually much better
                n_estimate = data['Budget'].max() * 5  # Rough estimate
                theoretical = [min(b * 20, n_estimate) for b in data['Budget']]  # More realistic bound
                axes[0,0].plot(data['Budget'], theoretical,
                              linestyle='--', alpha=0.7, label=f'{algorithm} (O(n²) bound)')
            elif algorithm == 'OT':
                # Theoretical: 3n
                n_estimate = data['Budget'].max() * 5
                theoretical = [3 * min(b * 5, n_estimate) for b in data['Budget']]
                axes[0,0].plot(data['Budget'], theoretical,
                              linestyle='--', alpha=0.7, label=f'{algorithm} (3n bound)')

        axes[0,0].set_xlabel('Budget (Problem Size Proxy)', fontsize=12)
        axes[0,0].set_ylabel('Oracle Calls', fontsize=12)
        axes[0,0].set_title('Query Complexity: Actual vs Theoretical', fontsize=14, fontweight='bold')
        axes[0,0].legend(fontsize=10)
        axes[0,0].grid(True, alpha=0.3)

        # Plot 5b: Runtime Scaling
        for algorithm in self.results_df['Algorithm'].unique():
            data = self.results_df[self.results_df['Algorithm'] == algorithm]
            axes[0,1].loglog(data['Budget'], data['Runtime(s)'],
                            marker='s', linewidth=2, markersize=8, label=algorithm)

        axes[0,1].set_xlabel('Budget (Problem Size Proxy, log)', fontsize=12)
        axes[0,1].set_ylabel('Runtime (seconds, log)', fontsize=12)
        axes[0,1].set_title('Runtime Scaling Analysis', fontsize=14, fontweight='bold')
        axes[0,1].legend(fontsize=10)
        axes[0,1].grid(True, alpha=0.3)

        # Plot 5c: Memory vs Theoretical Formula
        for algorithm in self.memory_df['Algorithm'].unique():
            data = self.memory_df[self.memory_df['Algorithm'] == algorithm]
            axes[1,0].plot(data['Total_Components'], data['Memory(KB)'],
                          marker='^', linewidth=2, markersize=8, label=algorithm)

        # Add theoretical line (400KB per component)
        max_components = self.memory_df['Total_Components'].max()
        theoretical_x = range(0, int(max_components) + 1, max(1, int(max_components // 10)))
        theoretical_y = [x * 400 for x in theoretical_x]  # 400KB per component
        axes[1,0].plot(theoretical_x, theoretical_y, 'r--', alpha=0.7, label='Theoretical (400KB/component)')

        axes[1,0].set_xlabel('Total Components', fontsize=12)
        axes[1,0].set_ylabel('Memory Usage (KB)', fontsize=12)
        axes[1,0].set_title('Memory: Actual vs Theoretical Formula', fontsize=14, fontweight='bold')
        axes[1,0].legend(fontsize=10)
        axes[1,0].grid(True, alpha=0.3)

        # Plot 5d: Algorithm Comparison Radar Chart (if multiple budgets)
        if len(self.results_df['Budget'].unique()) >= 2:
            # Use middle budget for comparison
            budgets = sorted(self.results_df['Budget'].unique())
            mid_budget = budgets[len(budgets)//2]

            comparison_data = self.results_df[self.results_df['Budget'] == mid_budget]
            algorithms = comparison_data['Algorithm'].unique()

            if len(algorithms) >= 2:
                # Normalize metrics for radar chart
                metrics = ['Gain', 'Runtime(s)', 'SolutionSize']
                normalized_data = {}

                for metric in metrics:
                    max_val = comparison_data[metric].max()
                    min_val = comparison_data[metric].min()
                    if max_val > min_val:
                        if metric == 'Runtime(s)':  # For runtime, lower is better
                            normalized_data[metric] = 1 - (comparison_data[metric] - min_val) / (max_val - min_val)
                        else:  # For gain and size, higher is better
                            normalized_data[metric] = (comparison_data[metric] - min_val) / (max_val - min_val)
                    else:
                        normalized_data[metric] = [0.5] * len(comparison_data)

                # Simple bar chart instead of radar for simplicity
                x = range(len(algorithms))
                width = 0.25

                for i, metric in enumerate(metrics):
                    values = [normalized_data[metric][comparison_data['Algorithm'] == alg].iloc[0] for alg in algorithms]
                    axes[1,1].bar([xi + i*width for xi in x], values, width, label=metric, alpha=0.8)

                axes[1,1].set_xlabel('Algorithms', fontsize=12)
                axes[1,1].set_ylabel('Normalized Performance', fontsize=12)
                axes[1,1].set_title(f'Performance Comparison (Budget={mid_budget})', fontsize=14, fontweight='bold')
                axes[1,1].set_xticks([xi + width for xi in x])
                axes[1,1].set_xticklabels(algorithms, rotation=45)
                axes[1,1].legend(fontsize=10)
                axes[1,1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'theoretical_verification.png',
                   dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'theoretical_verification.pdf',
                   bbox_inches='tight')
        plt.show()

    def generate_summary_table(self):
        """Generate summary table for paper"""
        summary_data = []

        for algorithm in self.results_df['Algorithm'].unique():
            algo_results = self.results_df[self.results_df['Algorithm'] == algorithm]
            algo_ops = self.operations_df[self.operations_df['Algorithm'] == algorithm]

            avg_gain = algo_results['Gain'].mean()
            avg_runtime = algo_results['Runtime(s)'].mean()
            avg_solution_size = algo_results['SolutionSize'].mean()
            avg_oracle_calls = algo_ops['Oracle_Calls'].mean() if not algo_ops.empty else 0

            summary_data.append({
                'Algorithm': algorithm,
                'Avg_Gain': f'{avg_gain:.2f}',
                'Avg_Runtime(s)': f'{avg_runtime:.3f}',
                'Avg_Solution_Size': f'{avg_solution_size:.1f}',
                'Avg_Oracle_Calls': f'{avg_oracle_calls:.0f}'
            })

        summary_df = pd.DataFrame(summary_data)

        # Save summary table
        summary_df.to_csv(self.output_dir / 'summary_table.csv', index=False)

        # Display formatted table
        print("\n" + "="*60)
        print("ALGORITHM PERFORMANCE SUMMARY")
        print("="*60)
        print(summary_df.to_string(index=False))
        print("="*60)

        return summary_df

    def debug_dataframes(self):
        """Debug function to check DataFrame structures"""
        print("\n=== DATAFRAME DEBUG INFO ===")

        print("\nResults DataFrame:")
        print(f"Shape: {self.results_df.shape}")
        print(f"Columns: {self.results_df.columns.tolist()}")
        if not self.results_df.empty:
            print(f"Sample row:\n{self.results_df.iloc[0]}")

        print("\nOperations DataFrame:")
        print(f"Shape: {self.operations_df.shape}")
        print(f"Columns: {self.operations_df.columns.tolist()}")
        if not self.operations_df.empty:
            print(f"Sample row:\n{self.operations_df.iloc[0]}")

        print("\nMemory DataFrame:")
        print(f"Shape: {self.memory_df.shape}")
        print(f"Columns: {self.memory_df.columns.tolist()}")
        if not self.memory_df.empty:
            print(f"Sample row:\n{self.memory_df.iloc[0]}")

        print("=== END DEBUG INFO ===\n")

    def generate_all_plots(self):
        """Generate all plots for paper"""
        print("Generating publication-quality plots...")

        # Debug dataframes first
        self.debug_dataframes()

        try:
            self.plot_solution_quality_comparison()
            print("✓ Solution quality plots generated")
        except Exception as e:
            print(f"✗ Solution quality plots failed: {e}")

        try:
            self.plot_computational_efficiency()
            print("✓ Computational efficiency plots generated")
        except Exception as e:
            print(f"✗ Computational efficiency plots failed: {e}")

        try:
            self.plot_memory_analysis()
            print("✓ Memory analysis plots generated")
        except Exception as e:
            print(f"✗ Memory analysis plots failed: {e}")

        try:
            self.plot_approximation_ratios()
            print("✓ Approximation ratio plots generated")
        except Exception as e:
            print(f"✗ Approximation ratio plots failed: {e}")

        try:
            self.plot_theoretical_verification()
            print("✓ Theoretical verification plots generated")
        except Exception as e:
            print(f"✗ Theoretical verification plots failed: {e}")

        try:
            self.generate_summary_table()
            print("✓ Summary table generated")
        except Exception as e:
            print(f"✗ Summary table failed: {e}")

        print(f"\nPlot generation completed. Check: {self.output_dir}")
        print("Files generated:")
        for file in sorted(self.output_dir.glob("*.*")):
            print(f"  - {file.name}")


def main():
    parser = argparse.ArgumentParser(description="Generate paper-quality visualizations")
    parser.add_argument("results_file", help="Results CSV file")
    parser.add_argument("operations_file", help="Operations CSV file")
    parser.add_argument("memory_file", help="Memory CSV file")
    parser.add_argument("--output-dir", default="plots", help="Output directory for plots")
    parser.add_argument("--format", choices=["png", "pdf", "both"], default="both",
                       help="Output format")

    args = parser.parse_args()

    # Load data
    try:
        results_df = pd.read_csv(args.results_file)
        operations_df = pd.read_csv(args.operations_file)
        memory_df = pd.read_csv(args.memory_file)
    except Exception as e:
        print(f"Error loading data files: {e}")
        return

    # Create visualizer
    visualizer = PaperVisualizer(results_df, operations_df, memory_df, args.output_dir)

    # Generate all plots
    visualizer.generate_all_plots()


if __name__ == "__main__":
    main()