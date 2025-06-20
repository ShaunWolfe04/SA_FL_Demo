
# Goal: Template for Flower 1.17 to be compatible with survival analysis
# Assumes dataset is tabular
# To run: pip install -r requirements.txt



import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message=".*np.find_common_type is deprecated.*")

    from collections import OrderedDict
    from typing import List, Tuple

    import pandas as pd
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision.transforms as transforms
    from datasets.utils.logging import disable_progress_bar
    from torch.utils.data import DataLoader
    from sklearn.utils import shuffle



    import flwr
    from flwr.client import Client, ClientApp, NumPyClient
    from flwr.common import Metrics, Context
    from flwr.server import ServerApp, ServerConfig, ServerAppComponents
    from flwr.server.strategy import FedAvg
    from flwr.simulation import run_simulation
    from flwr_datasets import FederatedDataset

    from auton_survival.preprocessing import Preprocessor
    from auton_survival.datasets import load_dataset
    from auton_survival.datasets import load_dataset
    from auton_survival.models import dsm
    import scipy.integrate

# Define trapz manually using trapezoid
scipy.integrate.trapz = scipy.integrate.trapezoid

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Training on {DEVICE}")
print(f"Flower {flwr.__version__} / PyTorch {torch.__version__}")
disable_progress_bar()


# TODO: Global variables that can be resized
NUM_CLIENTS = 10 # The number of partitions
BATCH_SIZE = 32 # Self explanatory, how large is a batch

# TODO: call your dataset
# Example for SUPPORT in auton_survival

# (uses auton_survival)
outcomes, features = load_dataset(dataset='SUPPORT')
cat_feats = ['sex', 'dzgroup', 'dzclass', 'income', 'race', 'ca']
num_feats = ['age', 'num.co', 'meanbp', 'wblc', 'hrt', 'resp', 
             'temp', 'pafi', 'alb', 'bili', 'crea', 'sod', 'ph', 
             'glucose', 'bun', 'urine', 'adlp', 'adls']

preprocessor = Preprocessor(cat_feat_strat='ignore', num_feat_strat= 'mean') 
transformer = preprocessor.fit(features, cat_feats=cat_feats, num_feats=num_feats,
                                one_hot=True, fill_value=-1)
features = transformer.transform(features)

combined_df = pd.concat([features, outcomes], axis=1)
# Need to convert this to a file to pass into FederatedDataset
temp_csv_path = "support_dataset_temp.csv"
combined_df.to_csv(temp_csv_path, index=False)
fds = FederatedDataset(dataset="csv", partitioners={"train": NUM_CLIENTS}, data_files=temp_csv_path)
"""What to change here: data_files"""


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


class DSMModel():

    def __init__(self, lr, inputdim, k, layers=None, dist='Weibull',
               temp=1000., discount=1.0, optimizer='Adam',
               risks=1):
        self.model = dsm.dsm_torch.DeepSurvivalMachinesTorch(inputdim, k, layers=layers, dist=dist, 
                                                            temp=temp, discount=discount, optimizer=optimizer, 
                                                            risks=risks)

        self.model.double() # Change to float64
        self.optimizer = dsm.utilities.get_optimizer(self.model, lr)

    def pretrain(self, t_tr_tensor, e_tr_tensor, t_val_tensor, e_val_tensor ):
        # pretrain to find a good starting point for parameters
        # adapted from auton_survival
        premodel = dsm.utilities.pretrain_dsm(self.model, t_tr_tensor, e_tr_tensor, t_val_tensor, e_val_tensor)
        for r in range(self.model.risks):
            self.model.shape[str(r+1)].data.fill_(float(premodel.shape[str(r+1)]))
            self.model.scale[str(r+1)].data.fill_(float(premodel.scale[str(r+1)]))

"""train function for DSM"""
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

        # validation step
        val_loss=0
        for r in range(model.model.risks):
            val_loss += dsm.losses.conditional_loss(model.model,
                                                x_val,
                                                dsm.utilities._reshape_tensor_with_nans(t_val),
                                                dsm.utilities._reshape_tensor_with_nans(e_val),
                                                elbo=True,
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
    # print(f"Test Loss: {val_loss.item():.4f}")
    return val_loss


def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]



# Call load_datasets in client_fn
class FlowerClient(NumPyClient):
    def __init__(self, dsmbase: DSMModel, x_tr, t_tr, e_tr, x_val, t_val, e_val):

        self.dsm_model=dsmbase
        self.model=self.dsm_model.model
        self.x_tr = x_tr
        self.t_tr = t_tr
        self.e_tr = e_tr
        self.x_val = x_val
        self.t_val = t_val
        self.e_val = e_val

        self.dsm_model.pretrain(self.t_tr, self.e_tr, self.t_val, self.e_val)
    
    def get_parameters(self, config):
        return get_parameters(self.model)
    
    def fit(self, parameters, config):
        set_parameters(self.model, parameters)
        self.dsm_model = train(self.dsm_model, self.x_tr, self.t_tr, self.e_tr, self.x_val, self.t_val, self.e_val, n_iter=2, bs=16)
        return get_parameters(self.model), len(self.x_tr), {}
    
    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        loss = test(self.dsm_model, self.x_val, self.t_val, self.e_val)
        return float(loss), len(self.x_val), {}


def client_fn(context: Context) -> Client:
    """Create a Flower client representing a single organization."""

    # Load model


    # Load data (SUPPORT)
    partition_id = context.node_config["partition-id"]
    x_tr, t_tr, e_tr, x_val, t_val, e_val = load_datasets(partition_id=partition_id)
    inputdim = x_tr.shape[-1]
    net = DSMModel(1e-3, inputdim, 3, layers=[100, 100], dist='Weibull', temp=1000., discount=1.0, optimizer='Adam', risks=1)

    # Create a single Flower client representing a single organization
    # FlowerClient is a subclass of NumPyClient, so we need to call .to_client()
    # to convert it to a subclass of `flwr.client.Client`
    return FlowerClient(net, x_tr, t_tr, e_tr, x_val, t_val, e_val).to_client()



# Create the ClientApp
client = ClientApp(client_fn=client_fn)



# Create FedAvg strategy
strategy = FedAvg(
    fraction_fit=1.0,  # Sample 100% of available clients for training
    fraction_evaluate=0.5,  # Sample 50% of available clients for evaluation
    min_fit_clients=10,  # Never sample less than 10 clients for training
    min_evaluate_clients=5,  # Never sample less than 5 clients for evaluation
    min_available_clients=10,  # Wait until all 10 clients are available
)


def server_fn(context: Context) -> ServerAppComponents:
    """Construct components that set the ServerApp behaviour.

    You can use the settings in `context.run_config` to parameterize the
    construction of all elements (e.g the strategy or the number of rounds)
    wrapped in the returned ServerAppComponents object.
    """

    # Configure the server for 5 rounds of training
    config = ServerConfig(num_rounds=5)

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

# Output?

