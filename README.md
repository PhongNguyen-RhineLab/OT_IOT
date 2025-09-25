# Online Algorithms for Object Detection under Budget Allocation

This repository contains the implementation of online algorithms for object detection under budget constraints, as described in the paper "Online approximate algorithms for Object detection under Budget allocation"

## Overview

The project implements three main algorithms for selecting optimal image subregions under budget constraints:
- **Greedy Search (GS)**: Algorithm 4 from the paper
- **Online Threshold (OT)**: Online algorithm with threshold-based selection  
- **Improved Online Threshold (IOT)**: Enhanced version with ε-approximation

## Features

- **Multiple Dataset Support**: CIFAR-10, CIFAR-100, STL-10, MNIST, FashionMNIST
- **Flexible Cost/Gain Functions**: Both simple saliency-based and full 4-component submodular functions
- **Performance Tracking**: Memory usage, operation counts, and runtime analysis
- **Budget Constraint Testing**: Multiple budget levels for comprehensive evaluation
- **Automated Experiments**: Complete experimental pipeline with CSV output

## Project Structure

```
├── main_experiments.py      # Main experiment runner
├── paper_algorithms.py      # Core algorithm implementations (GS, OT, IOT)
├── dataset_loader.py        # Dataset loading and saliency map generation
├── image_division.py        # Algorithm ID - Image division into subregions
├── cost_gain_functions.py   # Cost and gain function implementations
├── operation_tracker.py     # Performance and operation tracking
├── memory_calculator.py     # Memory usage calculation
└── README.md                # This file
```

## Installation

### Prerequisites
- Python 3.7+
- PyTorch
- torchvision
- numpy
- pandas

### Setup
```bash
# Clone the repository
git clone <repository-url>
cd OT_IOT

# Install required packages
pip install torch torchvision numpy pandas
```

## Usage

### Basic Usage
Run experiments with default parameters:
```bash
python main_experiments.py
```

### Advanced Usage
Customize experiment parameters:
```bash
python main_experiments.py --dataset cifar10 --num-samples 50 --budgets 10 20 50 100 --algorithms greedy ot iot --verbose
```

### Parameters

| Parameter | Description | Default | Options |
|-----------|-------------|---------|---------|
| `--dataset` | Dataset to use | `cifar10` | `cifar10`, `cifar100`, `stl10`, `mnist`, `fashionmnist` |
| `--num-samples` | Number of images to process | `20` | Any positive integer |
| `--budgets` | Budget constraints to test | `[10, 20, 50]` | List of integers |
| `--algorithms` | Algorithms to run | `["greedy", "ot", "iot"]` | Subset of `["greedy", "ot", "iot"]` |
| `--epsilons` | Epsilon values for IOT | `[0.1, 0.2]` | List of floats |
| `--m` | Subregions per image | `8` | Positive integer |
| `--N` | Patch grid size (N×N) | `4` | Positive integer |
| `--use-submodular` | Use full submodular function | `False` | Flag |
| `--verbose` | Verbose output | `False` | Flag |

## Algorithms

### 1. Greedy Search (GS)
- **Paper Reference**: Algorithm 4
- **Type**: Offline algorithm
- **Approach**: Iteratively selects subregions with highest gain-to-cost ratio
- **Time Complexity**: O(n²) where n is the number of subregions

### 2. Online Threshold (OT) 
- **Type**: Online algorithm
- **Approach**: Uses threshold-based selection for streaming subregions
- **Advantage**: Processes data as it arrives without full dataset knowledge

### 3. Improved Online Threshold (IOT)
- **Type**: Online algorithm with ε-approximation
- **Approach**: Enhanced threshold strategy with approximation guarantees
- **Parameters**: Configurable ε values for approximation quality

## Cost and Gain Functions

### Cost Function
- **Type**: Modular cost function
- **Formula**: c(S) = Σ_{I^M ∈ S} c(I^M)
- **Implementation**: Based on subregion area with proper scaling

### Gain Functions
1. **Simple Saliency-based**: Uses saliency map values directly
2. **4-Component Submodular**: Includes saliency, diversity, coverage, and model confidence

## Output

The experiments generate CSV files with the following metrics:
- **Algorithm**: Which algorithm was used
- **Budget**: Budget constraint value
- **Total_Gain**: Achieved gain value
- **Total_Cost**: Used budget/cost
- **Budget_Utilization**: Percentage of budget used
- **Runtime_seconds**: Execution time
- **Memory_MB**: Peak memory usage
- **Selected_Regions**: Number of selected subregions
- **Operations**: Detailed operation counts

## Performance Tracking

The system tracks:
- **Memory Usage**: Peak memory consumption during execution
- **Operation Counts**: Detailed breakdown of computational operations
- **Runtime**: Algorithm execution time
- **Budget Utilization**: Efficiency of budget usage

## Research Paper

This implementation is based on the research paper "Online approximate algorithms for Object detection under Budget allocation" submitted to SOICT 2025. The paper provides theoretical analysis and experimental validation of the proposed algorithms.

## Example Results

```
Algorithm: Greedy_GS, Budget: 20, Gain: 15.47, Cost: 19.8, Runtime: 0.23s
Algorithm: OT, Budget: 20, Gain: 14.12, Cost: 18.5, Runtime: 0.15s  
Algorithm: IOT_eps0.1, Budget: 20, Gain: 14.89, Cost: 19.2, Runtime: 0.18s
```

## Contributing

This is a research implementation. For questions or collaborations, please refer to the paper authors.

## License

See LICENSE file for details.


