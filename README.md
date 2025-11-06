# migration_decisions_bangladesh

This repository accompanies the article:
**"Combined models of violent conflict and natural hazards improve predictions of household mobility in Bangladesh"**

It provides the code, processed data, and environment specification needed to reproduce all main results and figures reported in the paper.

## Repository Structure
- `input_data/` — cleaned replication data used for the analysis 
- `results/` — prediction results, figures and tables from the paper  
- `environment_simple.yml` — Conda environment for replication  
- `main_analysis.ipynb` — Main analysis for replication
- `evaluate_train_combined.py` — Functions for training and predicting
- `plotting_functions.py` — Functions for plotting
- `functions_transforms.py` — Functions for transformations
- `variable_dict.json` — Dictonary with long variable names

## Reproducing the Analysis

1. **Create the environment**
   ```bash
   conda env create -f environment_simple.yml
   conda activate ccm_bangladesh
2. **Open and run the notebook**
   ```bash
   jupyter notebook main_analysis.ipynb
