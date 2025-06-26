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

--carbon (\<c\>): Carbon emission constraint level (1-4, representing different carbon policy scenarios)

--sediment (\<s\>): Minimum sediment transport requirement (14-54, with 0.2 increments)

--limit (\<l\>): Transmission line scenario (1, 2, 4, or 6 representing different infrastructure configurations)

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

### Reproducing Figures in the Manuscript
We have uploaded all the output files and code used to reproduce the figures in the manuscript in the folder `code_data_reproduce_figures`.

Please note that some original output data files in NetCDF (.nc) format are large in size, so they have been compressed to meet GitHub's size limitations. You can download and extract them locally to access the full data.

To view and process the .nc files, we recommend using the [xarray](https://docs.xarray.dev/) package.

### Reproducing Uncertainty Scenario Results
Before reproducing the results of the scenario analysis, you need to generate the inputs for the uncertainty scenarios first. Since the uncertainty analysis involves 72,900 scenarios, we did not upload all input files. Please follow the steps below to reproduce uncertainty scenario results:

1. Run `./input/generate_uncertainty_input.py` to generate the scenario inputs. You can modify the `start_no`, `end_no`, and `exclude_no` parameters in the script to flexibly define the range and specific scenarios you wish to generate.
2. After running the script of generating inputs, you will obtain a series of input files named `input_$no.xlsx` (where `no` depends on your specified settings), as well as a summary file named `uncertainty_scenarios_$start_no_$end_no.xlsx`. The summary file includes a `no` column, followed by the parameter values corresponding to the candidate inputs in the `input/uncertainty_input_data` folder.
3. Once the uncertainty inputs are generated, run the following command to simulate one or multiple scenarios:
   ```bash
   python run_mekong_gurobi-uncertainty.py $start_no $end_no
   ```
4. To run multiple scenarios in batch mode on a supercomputer, use the `uncertainty_script_55289_61964.sh` script.
   - This script is designed for [Aspire2a](https://www.nscc.sg/aspire-2a/), which uses the PBS job scheduler. You will need to modify the script according to your computing platform.
5. After running the scripts, results will be saved under the `./output/Uncertainty` directory.

**Additional Resources:**
- Summary of all inputs: `input/uncertainty_all_scenarios.xlsx`
- All output data for uncertainty scenarios is available at: [https://zenodo.org/records/15679035](https://zenodo.org/records/15679035)

## Contact
For questions regarding the code or methodology, please contact [Zhanwei Liu](liuzhanwei@u.nus.edu).
