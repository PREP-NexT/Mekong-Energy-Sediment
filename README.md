# Mekong-Energy-Sediment

## Introduction

This repository contains data and code accompanying the manuscript:

**Xu et al., Strategizing Renewable Energy Transitions to Preserve Sediment Transport Integrity** (under review)

The study investigates optimal renewable energy deployment strategies in the Mekong River basin that balance carbon emission reduction targets with the preservation of critical sediment transport processes.

## Requirements

### Core Environment
- **Python 3.10.16** (Recommended via [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- **Gurobi Optimizer** (with academic/commercial license)

### Installation Options

#### Option 1: Pip

```bash
pip install numpy==1.26.4 pandas==2.0.3 scipy==1.11.4 pyomo==6.9.1 xarray==2023.6.0 gurobipy==12.0.1
```
#### Option 2: Conda (Recommended)

```bash
conda create -n mekong python=3.10.16
conda activate mekong
conda install -c conda-forge numpy=1.26.4 pandas=2.0.3 scipy=1.11.4 pyomo=6.9.1 xarray=2023.6.0
conda install -c gurobi gurobi
```

## Usage
### Single Run Execution
Run a single scenario with specified parameters:

```bash
python run_mekong_gurobi.py --carbon=<c> --sediment=<s> --limit=<l>
```

Where:

--carbon (<c>): Carbon emission constraint level (1-4, representing different carbon policy scenarios)

--sediment (\<s\>): Minimum sediment transport requirement (14-54, with 0.2 increments)

--limit (<l>): Transmission line scenario (1, 2, 4, or 6 representing different infrastructure configurations)

### Parallel Execution
For comprehensive parameter space exploration, use the provided parallel execution script:

```bash
./parallel.sh
```
The script implements a parallelized sweep through:

+ 4 transmission scenarios (limit)

+ 4 carbon policy levels (carbon)

+ 201 sediment transport constraints (sediment from 14 to 54 in 0.2 increments)

The `parallel.sh` script uses GNU parallel to run N=10 concurrent processes for efficient computation.

## Contact
For questions regarding the code or methodology, please contact [Zhanwei Liu] at [liuzhanwei@u.nus.edu]
