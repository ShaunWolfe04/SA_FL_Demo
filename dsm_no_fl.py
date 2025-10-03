import pandas as pd
import sys
sys.path.append('../')

import scipy.integrate

# Define trapz manually using trapezoid
scipy.integrate.trapz = scipy.integrate.trapezoid

from tqdm import tqdm
import numpy as np
from copy import deepcopy
from auton_survival.datasets import load_dataset
from auton_survival.models import dsm
import torch
from sklearn.utils import shuffle



class DSMModel(dsm.DSMBase):

    def __init__(self, inputdim, k, layers=None, dist='Weibull',
               temp=1000., discount=1.0, random_seed=0
               # optimizer='Adam', risks=1
               ):
        super().__init__( k, layers, dist, temp, discount, random_seed)
        
        self.torch_model = None
        self.lr = None
        self.optimizer = None
        self.stop_training=False

    def setup_model(self, t_tr, e_tr, t_val, e_val, inputdim=None, x_tr = None, lr=1e-3, optimizer='Adam', risks=1):
        if inputdim is None:
            assert x_tr is not None, "one of x_tr or inputdim must be provided"
            inputdim = x_tr.shape[-1]
        
        self.torch_model = self._gen_torch_model(inputdim, optimizer, risks)
        self.torch_model.double()

        self._pretrain(t_tr, e_tr, t_val, e_val)

        self.optimizer=optimizer
        self._setup_optimizer(lr)

    def _pretrain(self, t_tr_tensor, e_tr_tensor, t_val_tensor, e_val_tensor ):
        # pretrain to find a good starting point for parameters
        # adapted from auton_survival
        
        premodel = dsm.utilities.pretrain_dsm(self.torch_model, t_tr_tensor, e_tr_tensor, t_val_tensor, e_val_tensor)
        for r in range(self.torch_model.risks):
            self.torch_model.shape[str(r+1)].data.fill_(float(premodel.shape[str(r+1)]))
            self.torch_model.scale[str(r+1)].data.fill_(float(premodel.scale[str(r+1)]))
        
    def _setup_optimizer(self, lr):
        # setup optimizer after pretraining and before training
        self.optimizer = dsm.utilities.get_optimizer(self.torch_model, lr)
    

def train(model: DSMModel, x_tr, t_tr, e_tr, x_val, y_val, n_iter=10, elbo=True, bs=100, random_seed=0, patience_limit=5):
    #TODO: Implement early stopping

    y_tr = pd.DataFrame({
    'time': t_tr.cpu().numpy(),
    'event': e_tr.cpu().numpy()
    })
    if model.stop_training is True:
        print("early stopping criterion met. Skipping train cycle...")
        return model
    nbatches = int(x_tr.shape[0]/bs)+1
    dics = []
    c_indices = []

    patience = 0
    best_c_index = -float('inf')
    
    torch.manual_seed(model.random_seed)
    np.random.seed(model.random_seed)


    for i in tqdm(range(n_iter)):
        x_tr, t_tr, e_tr = shuffle(x_tr, t_tr, e_tr, random_state=i)
        for j in range(nbatches):
            
            xb=x_tr[j*bs:(j+1)*bs]
            tb=t_tr[j*bs:(j+1)*bs]
            eb=e_tr[j*bs:(j+1)*bs]
            if xb.shape[0] == 0:
                continue

            #resets the optimizer so it can run for this loop
            model.optimizer.zero_grad()
            loss=0
            for r in range(model.torch_model.risks):
                # complex loss function of the dsm model needs to be outsourced
                loss += dsm.losses.conditional_loss(model.torch_model,
                                                    xb,
                                                    tb,
                                                    eb,
                                                    elbo=elbo,
                                                    risk=str(r+1))
            # print ("Train Loss", float(loss))
            loss.backward()
            
            model.optimizer.step()



        current_c_index = evaluate_model(model, x_val, y_val, y_tr)
        c_indices.append(current_c_index)
        dics.append(deepcopy(model.torch_model.state_dict()))
        
        #print(f"Epoch {i+1}/{n_iter} - Validation C-Index: {current_c_index:.4f}")

        # Check for improvement
        if current_c_index > best_c_index:
            best_c_index = current_c_index
            patience = 0 # Reset patience because we found a better model
            #print(f"  New best C-Index found: {best_c_index:.4f}")
        else:
            patience += 1 # No improvement
            #print(f"  No improvement in C-Index for {patience} epoch(s).")
        
        if patience >= patience_limit:
            #print(f"\nEarly stopping triggered after {patience} epochs with no improvement.")
            break

    # --- Revert to the best model found ---
    print(f"\nTraining finished. Best Validation C-Index: {max(c_indices):.4f}")
    best_epoch_idx = np.argmax(c_indices)
    model.torch_model.load_state_dict(dics[best_epoch_idx])
    print(f"Model state from epoch {best_epoch_idx + 1} loaded.")

    del dics
    model.fitted = True
    return model



def test(model, x_te, t_te, e_te):
        # validation step
    model.torch_model.eval() # Set the model to evaluation mode (turns off dropout, etc.)
    test_loss=0
    with torch.no_grad(): # Disable gradient calculations for speed and memory
        for r in range(model.torch_model.risks):
            test_loss += dsm.losses.conditional_loss(model.torch_model,
                                                x_te,
                                                t_te,
                                                e_te,
                                                elbo=False,
                                                risk=str(r+1))
    print(f"Test Loss: {test_loss.item():.4f}")

    return test_loss


from auton_survival.estimators import _predict_dsm
from auton_survival.metrics import survival_regression_metric
import numpy as np

# evalutate_model made by Gemini
def evaluate_model(model, x_eval, y_eval, y_train):
    """
    Evaluates the DSM model on a given dataset and returns the Concordance Index.

    Parameters:
    - model: The DSMModel object.
    - x_eval (pd.DataFrame): Features of the evaluation set.
    - y_eval (pd.DataFrame): Outcomes (time, event) of the evaluation set.
    - y_train (pd.DataFrame): Outcomes of the training set, used for defining time bins.

    Returns:
    - float: The average Concordance Index.
    """
    # Ensure the model is in evaluation mode for predictions
    model.torch_model.eval()
    
    # Define the time points for evaluation based on the training set outcomes
    times = np.quantile(y_train['time'][y_train['event']==1], np.linspace(0.1, 0.9, 10)).tolist()

    # Get model predictions
    # Note: _predict_dsm requires the model to have .fitted = True
    model.fitted = True 
    predictions = _predict_dsm(model, x_eval, times)
    
    # Calculate Concordance Index
    # We use np.mean() because the function returns scores for each time point
    c_index = np.mean(survival_regression_metric('ctd', outcomes=y_eval, predictions=predictions, 
                                                 times=times, outcomes_train=y_train))
    
    # Set the model back to training mode
    model.torch_model.train()
    
    return c_index


#code from Gemini to inspect gradients
#used for debugging
def print_grad_stats(name):
    """A helper function to create our hook."""
    def hook(grad):
        if grad is not None:
            print(f"--- Gradients for '{name}' ---")
            print(f"  - Shape: {grad.shape}")
            print(f"  - Mean:  {grad.mean():.4e}") # e-notation for large/small numbers
            print(f"  - Std:   {grad.std():.4e}")
            print(f"  - Max:   {grad.max():.4e}")
            print(f"  - Norm:  {grad.norm():.4e}") # The overall magnitude of the gradients
    return hook

#code from Gemini to view parameters
#used for debugging
def view_params(model):
    pytorch_model = model

    print("--- Inspecting Model Weights using state_dict() ---")

    # 1. Get the state dictionary from the PyTorch model
    state_dict = pytorch_model.state_dict()

    # 2. Loop through the items in the dictionary to print them nicely
    for param_name, param_tensor in state_dict.items():
        
        print(f"Parameter: '{param_name}'")
        print(f"  - Shape: {param_tensor.shape}")
        
        # To avoid printing huge tensors, let's just look at the first 5 values
        # We flatten the tensor to easily grab the first few elements
        preview_values = param_tensor.flatten()[:5]
        
        # We round them to make them easier to read
        print(f"  - Preview of values: {preview_values.round(decimals=4).tolist()}")
        print("-" * 20)

import argparse 
def main():
    # --- Load the dataset --- #
    parser = argparse.ArgumentParser(description="Run centralized DSM experiments.")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=['SUPPORT', 'FRAMINGHAM', 'PBC'],
        default='SUPPORT',
        help="Name of the dataset to use."
    )
    args = parser.parse_args()
    outcomes, features = None, None
    if args.dataset == "SUPPORT":
        outcomes, features = load_dataset(dataset=args.dataset)
        fill_values = {
            'alb': 3.5,
            'pafi': 333.3,
            'bili': 1.01,
            'crea': 1.01,
            'bun': 6.51,
            'wblc': 9.0,
            'urine': 2502.0
        }

        features.fillna(value=fill_values, inplace=True)

    else:
        x, t, e = load_dataset(dataset=args.dataset)
        features = pd.DataFrame(x)
        outcomes = pd.DataFrame({'time': t, 'event':e })
    cat_feats = ['sex', 'dzgroup', 'dzclass', 'income', 'race', 'ca']
    num_feats = ['age', 'num.co', 'meanbp', 'wblc', 'hrt', 'resp', 
                'temp', 'pafi', 'alb', 'bili', 'crea', 'sod', 'ph', 
                'glucose', 'bun', 'urine', 'adlp', 'adls']




    # --- Preprocess data --- # 

    import numpy as np
    from sklearn.model_selection import train_test_split

    # Split the SUPPORT data into training, validation, and test data
    x_tr, x_te, y_tr, y_te = train_test_split(features, outcomes, test_size=0.2, random_state=1)
    x_tr, x_val, y_tr, y_val = train_test_split(x_tr, y_tr, test_size=0.25, random_state=1) 

    print(f'Number of training data points: {len(x_tr)}')
    print(f'Number of validation data points: {len(x_val)}')
    print(f'Number of test data points: {len(x_te)}')
    if args.dataset == "SUPPORT":
        from auton_survival.preprocessing import Preprocessor

        # Fit the imputer and scaler to the training data and transform the training, validation and test data
        preprocessor = Preprocessor(cat_feat_strat='mode', num_feat_strat= 'mean') 
        transformer = preprocessor.fit(features, cat_feats=cat_feats, num_feats=num_feats,
                                        one_hot=True, fill_value=-1)
        x_tr = transformer.transform(x_tr)
        x_val = transformer.transform(x_val)
        x_te = transformer.transform(x_te)

    print(x_tr.head(5))

    # --- Make data tensors in a stupid way probably --- #
    
    t_tr = y_tr['time'].values
    e_tr = y_tr['event'].values
    t_val = y_val['time'].values
    e_val = y_val['event'].values
    t_te = y_te['time'].values
    e_te = y_te['event'].values

    x_tr_tensor = torch.tensor(x_tr.values, dtype=torch.float64)
    t_tr_tensor = torch.tensor(t_tr, dtype=torch.float64)
    e_tr_tensor = torch.tensor(e_tr, dtype=torch.float64)

    x_val_tensor = torch.tensor(x_val.values, dtype=torch.float64)
    t_val_tensor = torch.tensor(t_val, dtype=torch.float64)
    e_val_tensor = torch.tensor(e_val, dtype=torch.float64)

    x_test_tensor = torch.tensor(x_te.values, dtype=torch.float64)
    t_test_tensor = torch.tensor(t_te, dtype=torch.float64)
    e_test_tensor = torch.tensor(e_te, dtype=torch.float64)    
    
    # --- Instantiate and Train the DSM Model ---
    
    import time

    inputdim = x_tr.shape[-1]
    #print(f"inputdim: {inputdim}")
    #change these parameters to train the model
    # view auton_survival.models.dsm.dsm_torch.py for more info

    param_grid = {
        'k': [1, 3],
        'layers': [[100], [100, 100]],
        'lr': [1e-3, 1e-4],
        'discount': [1.0, 0.5] # A DSM-specific regularization parameter
    }
    model = DSMModel( inputdim, 6, layers=[50, 50], dist='Weibull', temp=1., discount=0.5, random_seed=0)
    model.setup_model(t_tr_tensor, e_tr_tensor, t_val_tensor, e_val_tensor, inputdim=inputdim, lr=1e-3, optimizer='Adam')
    # DEBUG STUFF
    """
    print("\n--- DEBUG: YOUR CODE'S PRE-FLIGHT CHECK ---")
    print("--- MODEL STATE ---")
    print(model.model) # We print the inner torch model
    print("\n--- OPTIMIZER STATE ---")
    print(model.optimizer) # Print the optimizer from your wrapper
    print("\n--- DATA TENSOR STATE ---")
    print(f"x_train shape: {x_tr_tensor.shape}, dtype: {x_tr_tensor.dtype}")
    print(f"x_train mean: {x_tr_tensor.mean():.4f}, std: {x_tr_tensor.std():.4f}")
    print(f"t_train shape: {t_tr_tensor.shape}, dtype: {t_tr_tensor.dtype}")
    print(f"t_train mean: {t_tr_tensor.mean():.4f}, std: {t_tr_tensor.std():.4f}")
    print("\n--- DATA STATE ---")
    x_np = x_tr_tensor.numpy()
    pd.DataFrame(x_np).to_csv("data_dsm_no_fl.csv", index=False)
    print("Saved to data_dsm_no_fl.csv")
    print("--- END YOUR CODE'S CHECK ---\n")

    """
    # DEBUG STUFF 2
    """
    print("\n--- Attaching Gradient Hooks to Model Layers ---")

    # We loop through all named layers in the underlying torch model
    for name, layer in model.model.named_modules():
        # We only care about the Linear layers where the weights are
        if isinstance(layer, torch.nn.Linear):
            # Register the hook on the .weight and .bias parameters of the layer
            layer.weight.register_hook(print_grad_stats(f"{name}.weight"))
            if layer.bias is not None:
                layer.bias.register_hook(print_grad_stats(f"{name}.bias"))

    print("Hooks attached. Starting training...\n")
    """
    #print("params before one run")
    #view_params(model.model)
    global_shape_params = model.torch_model.shape['1'].data.cpu().numpy()
    global_scale_params = model.torch_model.scale['1'].data.cpu().numpy()

    print(f"\nFinal Aggregated Global Model Parameters (k={len(global_shape_params)}):")
    for i in range(len(global_shape_params)):
        print(f"  Distribution {i+1}: Shape = {global_shape_params[i]:.4f}, Scale = {global_scale_params[i]:.4f}")

    

    train(model, x_tr_tensor, t_tr_tensor, e_tr_tensor, x_val, y_val, n_iter = 100)

    global_shape_params = model.torch_model.shape['1'].data.cpu().numpy()
    global_scale_params = model.torch_model.scale['1'].data.cpu().numpy()

    print(f"\nFinal Aggregated Global Model Parameters (k={len(global_shape_params)}):")
    for i in range(len(global_shape_params)):
        print(f"  Distribution {i+1}: Shape = {global_shape_params[i]:.4f}, Scale = {global_scale_params[i]:.4f}")

    
    test(model, x_test_tensor, t_test_tensor, e_test_tensor)
    
    times = np.quantile(y_tr['time'][y_tr['event']==1], [0.25, 0.5, 0.75]).tolist()

    from auton_survival.estimators import _predict_dsm
    from auton_survival.metrics import survival_regression_metric
    predictions_te = _predict_dsm(model, x_te, times)


    results = dict()
    results['Brier Score'] = survival_regression_metric('brs', outcomes=y_te, predictions=predictions_te, 
                                                    times=times, outcomes_train=y_tr)

    results['Concordance Index'] = survival_regression_metric('ctd', outcomes=y_te, predictions=predictions_te, 
                                                    times=times, outcomes_train=y_tr)
    
    for metric, scores in results.items():
    # Print the overall average score for the metric, as before
        print(f"\nAverage {metric}: {np.mean(scores):.4f}")
        
        # Print the individual scores for each time point
        print(f"  Breakdown by Time Point:")
        if scores is not None and len(scores) > 0:
            for i, score in enumerate(scores):
                time_point = times[i]
                quantile = (i + 1) * 25
                print(f"    - At time {int(time_point)} ({quantile}th percentile): {score:.4f}")
        else:
            print("    - Scores could not be computed.")
    # from auton_survival tutorial
    from estimators_demo_utils import plot_performance_metrics
    plot_performance_metrics(results, times)
    # Results: pretrain creates a loss of roughly 4.7
    # train creates a loss of roughly 4.5
    # the pretrain absolutely carries
    # the features here must not be highly correlated to survival
if __name__ == '__main__':
    main()
