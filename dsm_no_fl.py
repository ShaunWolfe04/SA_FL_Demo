import pandas as pd
import sys
sys.path.append('../')

import scipy.integrate

# Define trapz manually using trapezoid
scipy.integrate.trapz = scipy.integrate.trapezoid


from auton_survival.datasets import load_dataset
from auton_survival.models import dsm
import torch

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

    x_tr_tensor = torch.tensor(x_tr.values, dtype=torch.float32)
    t_tr_tensor = torch.tensor(t_tr, dtype=torch.float32)
    e_tr_tensor = torch.tensor(e_tr, dtype=torch.float32)

    x_val_tensor = torch.tensor(x_val.values, dtype=torch.float32)
    t_val_tensor = torch.tensor(t_val, dtype=torch.float32)
    e_val_tensor = torch.tensor(e_val, dtype=torch.float32)
    
    # --- 2. Instantiate and Train the DSM Model ---
    
    print("\n--- Step 2: Training the Deep Survival Machines model ---")
    #TODO
    # Define a DeepSurvivalAnalysisTorch model
    # Do training loop (most is defined in auton_survival.models.dsm)
    # fit function can't be called since it is monolithic
    # must recreate fit function to run n epochs


    

    exit()
    model = dsm.utilities.pretrain_model()
   

if __name__ == '__main__':
    main()
