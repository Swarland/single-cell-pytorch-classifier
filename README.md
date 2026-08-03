## single-cell-pytorch-classifier
A PyTorch project for classifying cell types from single-cell RNA-seq data.

## Data

700 cells
765 genes
10 cell types

## Results

Test accuracy: 82.9%

Confusion matrix
Training curves
Classification report

## Repository Structure

src/
- dataset.py
- model.py
- train.py
- evaluate.py
- optuna.py
- utils.py

notebooks/
- 01_explore_data.ipynb
- 02_train_model.ipynb
- 03_final_evaluation.ipynb

data/
- raw datasets

reports/
- Quarto reports

figures/
- plots and model outputs