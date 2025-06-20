import pandas as pd
import sys
sys.path.append('../')

import scipy.integrate

# Define trapz manually using trapezoid
scipy.integrate.trapz = scipy.integrate.trapezoid


from auton_survival.datasets import load_dataset
from auton_survival.models import dsm
import torch
from sklearn.utils import shuffle



class DSMModel():

    def __init__(self, lr, inputdim, k, layers=None, dist='Weibull',
               temp=1000., discount=1.0, optimizer='Adam',
               risks=1):
        self.model = dsm.dsm_torch.DeepSurvivalMachinesTorch(inputdim, k, layers=layers, dist=dist, 
                                                            temp=temp, discount=discount, optimizer=optimizer, 
                                                            risks=risks)

        self.model.double() # Change to float64
        self.optimizer = dsm.utilities.get_optimizer(self.model, lr)

        self.patience = 0
        self.oldcost = float('inf')

        

        self.dics = []
        self.costs = []
    def pretrain(self, t_tr_tensor, e_tr_tensor, t_val_tensor, e_val_tensor ):
        # pretrain to find a good starting point for parameters
        # adapted from auton_survival
        premodel = dsm.utilities.pretrain_dsm(self.model, t_tr_tensor, e_tr_tensor, t_val_tensor, e_val_tensor)
        for r in range(self.model.risks):
            self.model.shape[str(r+1)].data.fill_(float(premodel.shape[str(r+1)]))
            self.model.scale[str(r+1)].data.fill_(float(premodel.scale[str(r+1)]))
    

def train(model: DSMModel, x_tr, t_tr, e_tr, x_val, t_val, e_val, n_iter=10, elbo=True, bs=100, random_seed=0):
    nbatches = int(x_tr.shape[0]/bs)+1
    # validation step
    # delete later
    
    


    for i in range(n_iter):
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
            for r in range(model.model.risks):
                loss += dsm.losses.conditional_loss(model.model,
                                                    xb,
                                                    dsm.utilities._reshape_tensor_with_nans(tb),
                                                    dsm.utilities._reshape_tensor_with_nans(eb),
                                                    elbo=elbo,
                                                    risk=str(r+1))
            # print ("Train Loss", float(loss))
            loss.backward()
            model.optimizer.step()

        # DEBUG
        shape_tensor = model.shape['1'] if hasattr(model, 'shape') else model.model.shape['1']
        
        # We access the first element of the tensor with .data[0] before calling .item()
        first_shape_value = shape_tensor.data[0].item()

        print(f"DEBUG: Epoch {i}, Batch {j}, First val of Shape Param '1': {first_shape_value:.6f}")

        # validation step
        val_loss=0
        for r in range(model.model.risks):
            val_loss += dsm.losses.conditional_loss(model.model,
                                                x_val,
                                                dsm.utilities._reshape_tensor_with_nans(t_val),
                                                dsm.utilities._reshape_tensor_with_nans(e_val),
                                                elbo=False,
                                                risk=str(r+1))
        print(f"Epoch {i+1}/{n_iter} | Val Loss: {val_loss.item():.4f}")
        
    return model


def test(model, x_te, t_te, e_te):
        # validation step
    model.model.eval() # Set the model to evaluation mode (turns off dropout, etc.)
    val_loss=0
    with torch.no_grad(): # Disable gradient calculations for speed and memory
        for r in range(model.model.risks):
            val_loss += dsm.losses.conditional_loss(model.model,
                                                x_te,
                                                dsm.utilities._reshape_tensor_with_nans(t_te),
                                                dsm.utilities._reshape_tensor_with_nans(e_te),
                                                elbo=True,
                                                risk=str(r+1))
    print(f"Test Loss: {val_loss.item():.4f}")
    
#code from Gemini to view parameters
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

def main():
    outcomes, features = load_dataset(dataset='SUPPORT')

    cat_feats = ['sex', 'dzgroup', 'dzclass', 'income', 'race', 'ca']
    num_feats = ['age', 'num.co', 'meanbp', 'wblc', 'hrt', 'resp', 
                'temp', 'pafi', 'alb', 'bili', 'crea', 'sod', 'ph', 
                'glucose', 'bun', 'urine', 'adlp', 'adls']




    # Data preprocessing

    import numpy as np
    from sklearn.model_selection import train_test_split

    # Split the SUPPORT data into training, validation, and test data
    x_tr, x_te, y_tr, y_te = train_test_split(features, outcomes, test_size=0.2, random_state=1)
    x_tr, x_val, y_tr, y_val = train_test_split(x_tr, y_tr, test_size=0.25, random_state=1) 

    print(f'Number of training data points: {len(x_tr)}')
    print(f'Number of validation data points: {len(x_val)}')
    print(f'Number of test data points: {len(x_te)}')
    from auton_survival.preprocessing import Preprocessor

    # Fit the imputer and scaler to the training data and transform the training, validation and test data
    preprocessor = Preprocessor(cat_feat_strat='ignore', num_feat_strat= 'mean') 
    transformer = preprocessor.fit(features, cat_feats=cat_feats, num_feats=num_feats,
                                    one_hot=True, fill_value=-1)
    x_tr = transformer.transform(x_tr)
    x_val = transformer.transform(x_val)
    x_te = transformer.transform(x_te)

    print(x_tr.head(5))

    
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
    
    # --- 2. Instantiate and Train the DSM Model ---
    
    print("\n--- Step 2: Training the Deep Survival Machines model ---")
    #TODO
    # Define a DeepSurvivalAnalysisTorch model
    # Do training loop (most is defined in auton_survival.models.dsm)
    # fit function can't be called since it is monolithic
    # must recreate fit function to run n epochs
    inputdim = x_tr.shape[-1]
    print(f"inputdim: {inputdim}")
    #change these parameters to train the model
    model = DSMModel(1e-3, inputdim, 3, layers=[100, 100], dist='Weibull', temp=1000., discount=1.0, optimizer='Adam', risks=1)
    model.pretrain(t_tr_tensor, e_tr_tensor, t_val_tensor, e_val_tensor)

    print("params before one run")
    view_params(model.model)

    train(model, x_tr_tensor, t_tr_tensor, e_tr_tensor, x_val_tensor, t_val_tensor, e_val_tensor, n_iter = 50)
    print("params after one run")
    view_params(model.model)
    print("Worked? ")
    
    test(model, x_test_tensor, t_test_tensor, e_test_tensor)

    exit()
    model = dsm.utilities.pretrain_model()
   

if __name__ == '__main__':
    main()
