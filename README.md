# Online Approximation Algorithms for Object Detection under Budget Allocation

This repository implements online approximation algorithms for submodular maximization under budget constraints, specifically designed for object detection tasks. The project includes three main algorithms: Greedy Search, OT (Online Threshold), and IOT (Improved Online Threshold) algorithms.

## Overview

The project addresses the problem of selecting optimal image subregions for object detection under computational budget constraints. It uses submodular optimization techniques to maximize detection performance while staying within memory and computational limits.

## Features

- **Three Optimization Algorithms**:
  - Greedy Search: Classic greedy approach for submodular maximization
  - OT Algorithm: Ultra-fast online threshold algorithm
  - IOT Algorithm: Improved online threshold with better approximation guarantees

- **Advanced Saliency Detection**:
  - Grad-CAM integration for generating attention maps
  - Support for various CNN architectures (ResNet18 by default)

- **Comprehensive Submodular Function**:
  - Confidence score using Evidential Deep Learning
  - Effectiveness measurement
  - Consistency evaluation
  - Collaboration assessment

- **Multi-Dataset Support**:
  - CIFAR-10/100
  - STL-10
  - MNIST
  - Fashion-MNIST
  - ImageNet (custom path)

- **Performance Monitoring**:
  - Runtime benchmarking
  - Theoretical memory usage calculation
  - Comparative analysis between algorithms

## Installation

### Requirements

```bash
pip install torch torchvision numpy pandas opencv-python scikit-learn
```

### Dependencies

- Python 3.8+
- PyTorch 1.9+
- torchvision
- numpy
- pandas
- opencv-python
- scikit-learn

## Project Structure

```
OT_IOT/
├── run_experiments.py      # Main experiment runner
├── greedy_search.py        # Greedy algorithm implementation
├── ot_algo.py             # Online Threshold algorithm
├── iot_algo.py            # Improved Online Threshold algorithm
├── gradcam.py             # Grad-CAM saliency generation
├── image_division.py      # Image patch division utilities
├── submodular_function.py # Submodular function implementation
└── README.md              # This file
```

## Usage

### Basic Usage

Run experiments with default parameters:

```bash
python run_experiments.py
```

### Advanced Usage

```bash
# Run on CIFAR-100 with custom parameters
python run_experiments.py --dataset cifar100 --num-samples 50 --budgets 1000 2000 4000

# Use full submodular function instead of simple saliency
python run_experiments.py --use-submodular --budgets 2000 4000 8000

# Test different epsilon values for IOT algorithm
python run_experiments.py --epsilons 0.1 0.2 0.5 --budgets 3000

# Custom number of subregions per image
python run_experiments.py --m 10 --budgets 5000
```

**Note**: Use this directory if run in Google Colab ```/content/OT_IOT/run_experiments.py```
### Command Line Arguments

- `--dataset`: Choose dataset (cifar10, cifar100, stl10, mnist, fashionmnist, imagenet)
- `--data-root`: Root directory for dataset storage
- `--num-samples`: Number of images to process
- `--budgets`: List of budget constraints to test
- `--epsilons`: List of epsilon values for IOT algorithm
- `--use-submodular`: Enable full submodular function (4 components)
- `--m`: Number of subregions per image

## Algorithms

### 1. Greedy Search
Classic greedy algorithm that iteratively selects the subregion with the highest gain-to-cost ratio.

**Time Complexity**: O(n²)  
**Memory**: O(n·m + |S|)

### 2. OT Algorithm (Online Threshold)
Ultra-fast online algorithm that uses density-based sorting and aggressive optimizations.

**Key Features**:
- Batch precomputation of singleton values
- Smart processing order based on gain density
- Early termination conditions

**Time Complexity**: O(n log n)  
**Memory**: O(m + |S+S'+1|)

### 3. IOT Algorithm (Improved Online Threshold)
Enhanced version of OT with better approximation guarantees using multiple threshold candidates.

**Key Features**:
- Multiple threshold testing
- Improved approximation ratio
- Memory-efficient candidate management

**Time Complexity**: O(n log n · |T|)  
**Memory**: O(m + |S+S'+1| + |max candidates|)

## Submodular Function

The project implements a comprehensive submodular function with four components:

```
g(S) = λ₁·s_conf + λ₂·s_eff + λ₃·s_cons + λ₄·s_colla
```

Where:
- **s_conf**: Confidence score using Evidential Deep Learning
- **s_eff**: Effectiveness based on prediction accuracy
- **s_cons**: Consistency across similar regions
- **s_colla**: Collaboration between different regions

## Output

The experiment generates a CSV file with results including:
- Algorithm name and parameters
- Dataset information
- Budget constraints
- Solution set size
- Objective function gain
- Runtime performance
- Memory usage

Example output:
```
Algorithm  Dataset  Budget  Epsilon  SetSize   Gain    Time(s)  Memory(KB)
Greedy     cifar10   2000      -        15     156.7     0.45      892.3
OT         cifar10   2000      -        14     151.2     0.12      445.6
IOT        cifar10   2000     0.1       14     153.8     0.23      567.8
```

## Performance

The algorithms are optimized for different scenarios:

- **Greedy**: Best solution quality, higher computational cost
- **OT**: Fastest execution, good approximation
- **IOT**: Balance between speed and solution quality

Typical performance on CIFAR-10 (100 images, budget=4000):
- Greedy: ~2.3s, best gain
- OT: ~0.5s, 95% of greedy gain
- IOT: ~1.1s, 98% of greedy gain

## Memory Efficiency

The project includes theoretical memory calculation based on:
- Number of images and subregions
- Algorithm-specific data structures
- Solution set sizes

Memory usage is optimized through:
- Lazy evaluation of submodular functions
- Efficient data structures
- Early pruning of infeasible candidates

## Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests and documentation
5. Submit a pull request

## License

This project is part of academic research. Please cite appropriately if used in publications.

## Citation

If you use this code in your research, please consider citing the associated paper.

## Contact

For questions or issues, please open a GitHub issue or contact me directly
