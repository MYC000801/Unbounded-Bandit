# Code for the paper "Improved Algorithms for Adversarial Bandits with Unbounded Losses"

Python version 3.7

Requires numpy, matplotlib, pandas

The data is contained in all_stocks_5yr.cvs and amazon_sales_dataset.csv, retrived from https://data.world/revanthkrishnaa/amazon-uk-sales-forecasting-2018-2021 and https://www.kaggle.com/datasets/camnugent/sandp500

Introduction: 

- algorithm.py implements the algorithms we test.
- data.py generates the loss/reward sequence from the dataset.
- test1.py includes the experiments on stock market data, amazon sales data and meta algorithm selection (corresponding to Figure 1, 2, 3).
- test2.py compares algorithms with extra exploration and non exploration (corresponding to Figure 4).
- test3.py compares UMAB-G with non-adaptive and adaptive exploration rate (corresponding to Figure 5, 6).

Instruction: run ''python main.py'' with different command scripts.

- python main.py stock: Figure 1.
- python main.py amazon: Figure 2.
- python main.py selection: Figure 3.
- python main.py exploration: Figure 4.
- python main.py adaptive: Figure 5.
- python main.py nonadaptive: Figure 6.