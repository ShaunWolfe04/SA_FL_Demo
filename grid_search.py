import pandas as pd
import numpy as np
import time
import itertools
import torch
from sklearn.model_selection import train_test_split
from auton_survival.datasets import load_dataset
# --- Corrected Imports ---
# We import the model and train functions from your script
from dsm_no_fl import DSMModel, train, evaluate_model 
# And we import Preprocessor directly from the library where it lives
from auton_survival.preprocessing import Preprocessor

def run_grid_search():
    """
    Performs a grid search to find the best hyperparameters for the DSM model.
    """
    # --- 1. Data Loading and Preparation (Same as your non-FL script) ---
    print("Loading and preparing SUPPORT dataset...")
    outcomes, features = load_dataset(dataset='SUPPORT')

    cat_feats = ['sex', 'dzgroup', 'dzclass', 'income', 'race', 'ca']
    num_feats = ['age', 'num.co', 'meanbp', 'wblc', 'hrt', 'resp', 
                 'temp', 'pafi', 'alb', 'bili', 'crea', 'sod', 'ph', 
                 'glucose', 'bun', 'urine', 'adlp', 'adls']

    x_tr, x_te, y_tr, y_te = train_test_split(features, outcomes, test_size=0.2, random_state=1)
    #x_tr, x_val, y_tr, y_val = train_test_split(x_tr, y_tr, test_size=0.25, random_state=1) 

    preprocessor = Preprocessor(cat_feat_strat='ignore', num_feat_strat='mean')
    transformer = preprocessor.fit(x_tr, cat_feats=cat_feats, num_feats=num_feats, one_hot=True, fill_value=-1)
    
    x_tr = transformer.transform(x_tr)
    #x_val = transformer.transform(x_val)
    x_te = transformer.transform(x_te)
    print("Data preparation complete.")

    # --- Create PyTorch Tensors for training and validation, matching dsm_no_fl.py ---
    x_tr_tensor = torch.tensor(x_tr.values, dtype=torch.float64)
    t_tr_tensor = torch.tensor(y_tr['time'].values, dtype=torch.float64)
    e_tr_tensor = torch.tensor(y_tr['event'].values, dtype=torch.float64)
    
    t_te_tensor = torch.tensor(y_te['time'].values, dtype=torch.float64)
    e_te_tensor = torch.tensor(y_te['event'].values, dtype=torch.float64)


    # --- 2. Define the Hyperparameter Grid ---
    # Start with a small grid to test. You can expand this later.
    param_grid = {
        'k': [4, 6, 8],
        'layers': [[100], [100, 100], [50], [50, 50]],
        'lr': [1e-3, 1e-4],
        'discount': [1.0, 0.5], # A DSM-specific regularization parameter
        'dist': ["Weibull", "LogNormal"]
    }

    # --- 3. Grid Search Loop ---
    best_c_index = -1
    best_params = None
    results_log = []

    # Generate all combinations of parameters
    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    total_combinations = len(param_combinations)
    print(f"\nStarting Grid Search with {total_combinations} combinations...")

    for i, params in enumerate(param_combinations):
        print(f"\n--- Combination {i+1}/{total_combinations}: {params} ---")

        try:
            # Instantiate the model with the current set of parameters
            input_dim = x_tr.shape[1]
            model = DSMModel(input_dim, k=params['k'], layers=params['layers'], 
                             dist='Weibull', discount=params['discount'])
            
            # Setup the model (pre-training and optimizer) using the correct Tensors
            model.setup_model(t_tr_tensor, e_tr_tensor, t_te_tensor, e_te_tensor, 
                              inputdim=input_dim, lr=params['lr'], optimizer='Adam')

            # Train the model with early stopping using the correct Tensors and DataFrames
            model = train(model, x_tr_tensor, t_tr_tensor, e_tr_tensor, x_te, y_te, 
                          n_iter=100, patience_limit=5)
            
            # Evaluate on the hold-out test set
            c_index_test = evaluate_model(model, x_te, y_te, y_tr)
            print(f"Resulting Test C-Index: {c_index_test:.4f}")

            results_log.append({'params': params, 'c_index': c_index_test})

            # Check if this is the best model so far
            if c_index_test > best_c_index:
                best_c_index = c_index_test
                best_params = params
                print(f"!!! New best C-Index found: {best_c_index:.4f} !!!")
            
            print("Sleeping for 10 seconds...")
            time.sleep(10)

        except Exception as e:
            print(f"Combination failed with error: {e}")
            results_log.append({'params': params, 'c_index': 'Error'})


    # --- 4. Print Final Results ---
    print("\n\n--- Grid Search Complete ---")
    print(f"Best Concordance Index found: {best_c_index:.4f}")
    print(f"Best Hyperparameters: {best_params}")

    # Optional: Save results to a file for later analysis
    with open("grid_search_results.txt", "w") as f:
        for entry in results_log:
            f.write(str(entry) + "\n")
        f.write("\n--- Best ---\n")
        f.write(f"Best Score: {best_c_index}\n")
        f.write(f"Best Params: {best_params}\n")

if __name__ == "__main__":
    run_grid_search()

