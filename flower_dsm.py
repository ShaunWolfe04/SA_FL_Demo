
# Goal: Template for Flower 1.17 to be compatible with survival analysis
# Assumes dataset is tabular
# To run: pip install -r requirements.txt



import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message=".*np.find_common_type is deprecated.*")
    from typing import Callable, Optional, Union
    from collections import OrderedDict
    from typing import List, Tuple

    import pandas as pd
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    import numpy as np
    import torch
    import torch.nn as nn
    from flwr.server.client_proxy import ClientProxy
    import torch.nn.functional as F
    import torchvision.transforms as transforms
    from datasets.utils.logging import disable_progress_bar
    from torch.utils.data import DataLoader
    from sklearn.utils import shuffle
    from copy import deepcopy



    import flwr as fl
    from flwr.client import Client, ClientApp, NumPyClient
    from flwr.common import Metrics, Context, Parameters, Scalar, FitRes
    from flwr.server import ServerApp, ServerConfig, ServerAppComponents
    from flwr.server.strategy import FedAvg
    from flwr.simulation import run_simulation
    from flwr_datasets import FederatedDataset

    from auton_survival.preprocessing import Preprocessor
    from auton_survival.datasets import load_dataset
    from auton_survival.models import dsm
    import scipy.integrate
    import sys
    scipy.integrate.trapz = scipy.integrate.trapezoid
    from auton_survival.estimators import _predict_dsm
    from auton_survival.metrics import survival_regression_metric

# Define trapz manually using trapezoid
import time
import os

start_time = time.time()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Training on {DEVICE}")
print(f"Flower {fl.__version__} / PyTorch {torch.__version__}")
disable_progress_bar()


# TODO: Global variables that can be resized
NUM_CLIENTS = 10 # The number of partitions
BATCH_SIZE = 32 # Self explanatory, how large is a batch

import argparse

# 1. Create the parser
parser = argparse.ArgumentParser(description="Run Federated Survival Analysis Experiments.")

# 2. Add arguments for your experimental variables
parser.add_argument(
    "--dataset", 
    type=str, 
    choices=['SUPPORT', 'METABRIC', 'SEER'], 
    default='SUPPORT', 
    help="Name of the dataset to use."
)
parser.add_argument(
    "--split", 
    type=str, 
    choices=['iid', 'dirichlet', 'location'], 
    default='iid', 
    help="Type of data split (iid or non-iid dirichlet)."
)
parser.add_argument(
    "--strategy", 
    type=str, 
    choices=['avg', 'prox'], 
    default='avg', 
    help="Federated learning strategy to use (FedAvg or FedProx)."
)

parser.add_argument(
    "--prox",
    type=float,
    default=0.1,
    help="Hyperparameter for FedProx, accessed when --strategy is \"prox\""
)

# 3. Parse the arguments
args = parser.parse_args()

from get_dataset import get_dataset

#13 clients for 13 states
if args.dataset=="SEER":
    NUM_CLIENTS = 13
x_tr, x_te, y_tr, y_te, fds= get_dataset(args.dataset, args.split, NUM_CLIENTS) 




print(f"FL with {NUM_CLIENTS} clients")

# (uses auton_survival)
"""
outcomes, features = load_dataset(dataset='SUPPORT')
cat_feats = ['sex', 'dzgroup', 'dzclass', 'income', 'race', 'ca']
num_feats = ['age', 'num.co', 'meanbp', 'wblc', 'hrt', 'resp', 
             'temp', 'pafi', 'alb', 'bili', 'crea', 'sod', 'ph', 
             'glucose', 'bun', 'urine', 'adlp', 'adls']

preprocessor = Preprocessor(cat_feat_strat='ignore', num_feat_strat= 'mean') 
transformer = preprocessor.fit(features, cat_feats=cat_feats, num_feats=num_feats,
                                one_hot=True, fill_value=-1)
features_upd = transformer.transform(features)


#this code is for direichlet
features_upd["dzgroup"]=features["dzgroup"]
from sklearn.model_selection import train_test_split
x_tr, x_te, y_tr, y_te = train_test_split(features_upd, outcomes, test_size=0.2, random_state=1)
"""


global_history = {
    "rounds": [],
    "Brier Score": [],
    "Concordance Index": []
}

def server_side_evaluation(server_round: int, parameters: fl.common.Parameters, config: dict):
    """
    This function is called by the strategy after aggregation.
    It evaluates the global model on the hold-out test set.
    """
    global global_history
    # 1. Use one of the client model objects as a template
    model = client_models[0]
    
    # 2. Load the aggregated weights into the model

    set_parameters(model.torch_model, parameters)
    model.fitted = True

    # 3. Define the time points for evaluation (should be consistent)
    times = np.quantile(y_tr['time'][y_tr['event']==1], np.linspace(0.1, 1, 10)).tolist()

    # 4. Perform prediction and calculate metrics

    predictions_te = _predict_dsm(model, x_te, times)
    results = dict()
    results['Brier Score'] = survival_regression_metric('brs', outcomes=y_te, predictions=predictions_te, 
                                                    times=times, outcomes_train=y_tr)

    results['Concordance Index'] = survival_regression_metric('ctd', outcomes=y_te, predictions=predictions_te, 
                                                    times=times, outcomes_train=y_tr)

    metrics_dict = {metric: float(np.mean(scores)) for metric, scores in results.items()}

    # --- FIX 2: Append the results to your global history object ---
    
    global_history["rounds"].append(server_round)
    global_history["Brier Score"].append(metrics_dict["Brier Score"])
    global_history["Concordance Index"].append(metrics_dict["Concordance Index"])
    # 5. Return the metric in a dictionary. Flower will log this.
    # We return the average C-Index over the time points.
    return None, metrics_dict


from strategy import gen_strategy
# handle args

config = {}
if args.strategy =='prox':
    config['proximal_mu']=0.1
strategy=gen_strategy(args.strategy, server_side_evaluation, **config)




def prepare_data(data):
    features = data.drop(['time', 'event'], axis=1)
    outcomes = data[['time', 'event']]
    x = features.values
    t = outcomes['time'].values
    e = outcomes['event'].values

    x_tensor = torch.tensor(x, dtype=torch.float64)
    t_tensor = torch.tensor(t, dtype=torch.float64)
    e_tensor = torch.tensor(e, dtype=torch.float64)

    return x_tensor, t_tensor, e_tensor


def load_datasets(partition_id: int):
    hf_partition= fds.load_partition(partition_id)

    if "_partition" in hf_partition.column_names:
        hf_partition = hf_partition.remove_columns("_partition")
    partition_train_test = hf_partition.train_test_split(test_size=0.2, seed=0)
    train = partition_train_test["train"].to_pandas()
    val = partition_train_test["test"].to_pandas()
    
    x_tr, t_tr, e_tr = prepare_data(train)
    x_val, t_val, e_val = prepare_data(val)

    return x_tr, t_tr, e_tr, x_val, t_val, e_val

# Check that it creates the correct datatype (Torch.tensor)
x_train, t_train, e_train, x_val, t_val, e_val = load_datasets(partition_id=0)
print(f"Type of x_train: {type(x_train)}")
print(x_train[:3, :5].round(decimals=2))


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
    

def train(model: DSMModel, x_tr, t_tr, e_tr, x_val, t_val, e_val, config=None, initial_parameters=None, n_iter=10, elbo=True, bs=100, random_seed=0):
 

    # --- CONFIG SETUP ---
    proximal_mu = config.get("proximal_mu") if config else None
    # If proximal_mu is still None at this point, set it to 0.0.
    proximal_mu = 0.0 if proximal_mu is None else proximal_mu
    # --- END SETUP ---

    if config is None:
        print("error")

    nbatches = int(x_tr.shape[0]/bs)+1
    dics = []
    costs = []
    valid_loss = test(model, x_val, t_val, e_val)
    
    valid_loss = valid_loss.detach().cpu().numpy()
    costs.append(float(valid_loss))
    dics.append(deepcopy(model.torch_model.state_dict()))  

    patience = 0
    oldcost = float('inf')
    
    #torch.manual_seed(model.random_seed)
    #np.random.seed(model.random_seed)


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
            for r in range(model.torch_model.risks):
                # complex loss function of the dsm model needs to be outsourced
                loss += dsm.losses.conditional_loss(model.torch_model,
                                                    xb,
                                                    tb,
                                                    eb,
                                                    elbo=elbo,
                                                    risk=str(r+1))
            # print ("Train Loss", float(loss))
            # --- FEDPROX MODIFICATION START ---
            if initial_parameters is not None and proximal_mu is not None:
                prox_term = 0.0
                # initial_parameters is a list of numpy arrays, convert to tensors
                global_params = [torch.from_numpy(p).to(DEVICE) for p in initial_parameters]
                
                # Iterate over current model parameters and the loaded global parameters
                for local_p, global_p in zip(model.torch_model.parameters(), global_params):
                    prox_term += (local_p - global_p).norm(2)**2
                    
                loss += (proximal_mu / 2) * prox_term
            # --- FEDPROX MODIFICATION END ---
            loss.backward()
            
            model.optimizer.step()



        valid_loss = test(model, x_val, t_val, e_val)
        valid_loss = valid_loss.detach().cpu().numpy()
        costs.append(float(valid_loss))
        dics.append(deepcopy(model.torch_model.state_dict()))

        # print(f"costs[-1]: {costs[-1]}, oldcost: {oldcost}")
        """
        if costs[-1] >= oldcost:
            if patience == 2: # or n_iter
                print("Model did not improve on this turn.")
                minm = np.argmin(costs)
                #model.torch_model.load_state_dict(dics[minm])

                del dics
                model.fitted = True
                model.stop_training = True
                return model
            else:
                patience += 1
        else:
            patience = 0
        oldcost = costs[-1]

    minm = np.argmin(costs)
    #model.torch_model.load_state_dict(dics[minm])

    del dics
    """

    model.fitted = True
    return model


def test(model, x_te, t_te, e_te):
        # validation step
    model.torch_model.eval() # Set the model to evaluation mode (turns off dropout, etc.)
    val_loss=0
    with torch.no_grad(): # Disable gradient calculations for speed and memory
        for r in range(model.torch_model.risks):
            val_loss += dsm.losses.conditional_loss(model.torch_model,
                                                x_te,
                                                t_te,
                                                e_te,
                                                elbo=False,
                                                risk=str(r+1))
    #print(f"Test Loss: {val_loss.item():.4f}")
    model.torch_model.train()
    return val_loss


def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v, dtype=torch.float64) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


client_models = []
for i in range(NUM_CLIENTS):
    print(f"Setting up model for client partition {i}...")
    
    # Load data for this client
    x_tr, t_tr, e_tr, x_val, t_val, e_val = load_datasets(partition_id=i)
    
    # Create and pre-train the model object
    input_dim = x_tr.shape[1]
    dsm_model = DSMModel(input_dim, 3, layers=[100, 100], dist='Weibull', temp=1000., discount=1.0, random_seed=0)
    dsm_model.setup_model(t_tr, e_tr, t_val, e_val, inputdim=input_dim, lr=1e-4, optimizer='Adam', risks=1)
    client_models.append(dsm_model)

    # Store the fully prepared, persistent model object
print("--- All client models are ready. ---\n")

# Call load_datasets in client_fn
class FlowerClient(NumPyClient):
    def __init__(self, dsmbase: DSMModel, x_tr, t_tr, e_tr, x_val, t_val, e_val):

        self.dsm_model=dsmbase
        
        self.x_tr = x_tr
        self.t_tr = t_tr
        self.e_tr = e_tr
        self.x_val = x_val
        self.t_val = t_val
        self.e_val = e_val

        
        self.model=self.dsm_model.torch_model
    def get_parameters(self, config):
        return get_parameters(self.model)
    
    def fit(self, parameters, config):
        set_parameters(self.model, parameters)
        self.dsm_model = train(self.dsm_model, self.x_tr, self.t_tr, self.e_tr, self.x_val, self.t_val, self.e_val, config, initial_parameters=parameters, n_iter=5, bs=16)
        return get_parameters(self.model), len(self.x_tr), {} # {} can be any sort of metrics
    
    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        loss = test(self.dsm_model, self.x_val, self.t_val, self.e_val)
        return float(loss), len(self.x_val), {}


def client_fn(context: Context) -> Client:
    """Create a Flower client representing a single organization."""

    # Load model
    warnings.filterwarnings("ignore", message=".*np.find_common_type is deprecated.*")

    # Load data (SUPPORT)
    partition_id = context.node_config["partition-id"]
    x_tr, t_tr, e_tr, x_val, t_val, e_val = load_datasets(partition_id=partition_id)
    inputdim = x_tr.shape[-1]
    net=client_models[partition_id]
    # Create a single Flower client representing a single organization
    # FlowerClient is a subclass of NumPyClient, so we need to call .to_client()
    # to convert it to a subclass of `flwr.client.Client`
    return FlowerClient(net, x_tr, t_tr, e_tr, x_val, t_val, e_val).to_client()



# Create the ClientApp
client = ClientApp(client_fn=client_fn)

# Create a NEW strategy (inherits from FedAvg)




def server_fn(context: Context) -> ServerAppComponents:
    """Construct components that set the ServerApp behaviour.

    You can use the settings in `context.run_config` to parameterize the
    construction of all elements (e.g the strategy or the number of rounds)
    wrapped in the returned ServerAppComponents object.
    """

    # Configure the server for 10 rounds of training
    config = ServerConfig(num_rounds=50)

    return ServerAppComponents(strategy=strategy, config=config)


# Create the ServerApp
server = ServerApp(server_fn=server_fn)


# Specify the resources each of your clients need
# By default, each client will be allocated 1x CPU and 0x GPUs
backend_config = {"client_resources": {"num_cpus": 1, "num_gpus": 0.0}}

# When running on GPU, assign an entire GPU for each client
if DEVICE == "cuda":
    backend_config = {"client_resources": {"num_cpus": 1, "num_gpus": 1.0}}
    # Refer to our Flower framework documentation for more details about Flower simulations
    # and how to set up the `backend_config`



# Run simulation
run_simulation(
    server_app=server,
    client_app=client,
    num_supernodes=NUM_CLIENTS, 
    backend_config=backend_config,
)
print("success")

print(global_history)

# here is where the code starts to be bad

weights_filepath = "round-50-weights.npz"

weights = np.load(weights_filepath)
parameters_list = [weights[key] for key in weights.files]
model = client_models[0]

set_parameters(model.torch_model, parameters_list)
print("params loaded back successfully")
model.fitted=True
times = np.quantile(y_tr['time'][y_tr['event']==1], np.linspace(0.1, 1, 10)).tolist()

from auton_survival.estimators import _predict_dsm
from auton_survival.metrics import survival_regression_metric
predictions_te = _predict_dsm(model, x_te, times)

"""
results = dict()
results['Brier Score'] = survival_regression_metric('brs', outcomes=y_te, predictions=predictions_te, 
                                                times=times, outcomes_train=y_tr)

results['Concordance Index'] = survival_regression_metric('ctd', outcomes=y_te, predictions=predictions_te, 
                                                times=times, outcomes_train=y_tr)

print('\n'.join([f"Average {metric}: {np.mean(scores):.4f}" for metric, scores in results.items()]))
"""


# Time to print results

end_time = time.time()
duration = end_time-start_time
print(f"Time to run experiment: {duration}")
experiment_name = f"{args.dataset}_{args.split}_{args.strategy}"
if args.strategy == 'prox':
    experiment_name += f"_mu{args.prox_mu}"

os.makedirs("plots", exist_ok=True)
os.makedirs("results", exist_ok=True)

if global_history["rounds"]:
    # For Brier Score, lower is better, so we use argmin
    best_brier_idx = np.argmin(global_history["Brier Score"])
    best_brier_round = global_history["rounds"][best_brier_idx]
    best_brier_score = global_history["Brier Score"][best_brier_idx]

    # For Concordance Index, higher is better, so we use argmax
    best_cindex_idx = np.argmax(global_history["Concordance Index"])
    best_cindex_round = global_history["rounds"][best_cindex_idx]
    best_cindex_score = global_history["Concordance Index"][best_cindex_idx]

    # Print the final summary
    print("\n--- Best Performance Across All Rounds ---")
    print(f"Round {best_brier_round}: Brier Score: {best_brier_score:.4f}")
    print(f"Round {best_cindex_round}: Concordance Index: {best_cindex_score:.4f}")

    # Add metrics to results file
    output_string = (
        f"Test: {experiment_name}\n"
        f"Round {best_brier_round}: Brier Score: {best_brier_score:.4f}\n"
        f"Round {best_cindex_round}: Concordance Index: {best_cindex_score:.4f}\n"
        f"Duration (seconds): {duration:.2f}\n"
        f"\n-------------------------------------------------\n"
    )

    # Append the results to the file
    results_path = os.path.join("results", "results.txt")
    with open(results_path, "a") as f:
        f.write(output_string)
    print(f"Best metrics appended to {results_path}")

    plt.figure(figsize=(10, 6))
    plt.plot(global_history["rounds"], global_history["Concordance Index"], marker='o', label="Concordance Index")
    plt.plot(global_history["rounds"], global_history["Brier Score"], marker='x', label="Brier Score")
    
    plt.title(f"Convergence: {experiment_name}")
    plt.xlabel("Communication Round")
    plt.ylabel("Metric Value")
    plt.grid(True)
    plt.legend()
    
    plot_path = os.path.join("plots", f"{experiment_name}_convergence.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Plotting was a success")
# from auton_survival tutorial
#from estimators_demo_utils import plot_performance_metrics
#plot_performance_metrics(results, times)

# Output?

