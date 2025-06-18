
# Goal: Template for Flower 1.17 to be compatible with survival analysis
# Assumes dataset is tabular
# To run: pip install -r requirements.txt
# Everywhere with TODO is something that can be customized


"""Now that we have all dependencies installed, we can import everything we need for this tutorial:"""
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message=".*np.find_common_type is deprecated.*")

    from collections import OrderedDict
    from typing import List, Tuple

    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision.transforms as transforms
    from datasets.utils.logging import disable_progress_bar
    from torch.utils.data import DataLoader



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
combined_df = pd.concat([features, outcomes], axis=1)
# Need to convert this to a file to pass into FederatedDataset
temp_csv_path = "support_dataset_temp.csv"
combined_df.to_csv(temp_csv_path, index=False)
fds = FederatedDataset(dataset="csv", partitioners={"train": NUM_CLIENTS}, data_files=temp_csv_path)
"""What to change here: data_files"""
#fds = FederatedDataset(dataset="cifar10", partitioners={"train": NUM_CLIENTS})

""" Code to test that it worked (Prints the head of one partition)
# Let's load a partition to prove it worked
client_partition = fds.load_partition(0, "train")
print("\nSuccessfully loaded data for client 0:")
print(client_partition.to_pandas().head())
# exit()
"""

""" data_preparer
TODO: make this work with your own dataset 
    (will involve transforming your own categorical features mostly)
Takes a pandas dataframe, either your train, validation, or test sets.
returns normalized tensors of features, event, time
Depending on the implementation, may need to return event and time together

There are three options on how to normalize numerical data:
1. No Scaling: don't normalize it at all
2. Local Scaling: normalize each partition independently (method that is implemented)
3. Federated Standardization: Get count, sum, and sum of squares for each partition, 
    then normalize based on those metrics

"""
def prepare_data(data, transformer):
    features = data.drop(['time', 'event'], axis=1)
    outcomes = data[['time', 'event']]
    x = transformer.transform(features).values
    t = outcomes['time'].values
    e = outcomes['event'].values

    x_tensor = torch.tensor(x, dtype=torch.float32)
    t_tensor = torch.tensor(t, dtype=torch.float32)
    e_tensor = torch.tensor(e, dtype=torch.float32)

    return x_tensor, t_tensor, e_tensor

"""   
Called once for each partition
Current structure returns: { training features, event, time, val features, event, time }
Depending on the implementation, may need to return event and time together
example given with SUPPORT
Calls prepare_data for repeated task
"""

def load_datasets(partition_id: int):
    hf_partition= fds.load_partition(partition_id)
    partition_train_test = hf_partition.train_test_split(test_size=0.2, seed=0)
    train = partition_train_test["train"].to_pandas()
    val = partition_train_test["test"].to_pandas()
    print("Columns in the 'train' DataFrame:", train.columns.to_list())
    features = train.drop(['time', 'event'], axis=1)
    cat_feats = ['sex', 'dzgroup', 'dzclass', 'income', 'race', 'ca']
    num_feats = ['age', 'num.co', 'meanbp', 'wblc', 'hrt', 'resp', 
             'temp', 'pafi', 'alb', 'bili', 'crea', 'sod', 'ph', 
             'glucose', 'bun', 'urine', 'adlp', 'adls']
    preprocessor = Preprocessor(cat_feat_strat='ignore', num_feat_strat='mean')
    transformer = preprocessor.fit(features, cat_feats=cat_feats, num_feats=num_feats,
                                    one_hot=True, fill_value=-1)
    
    x_tr, t_tr, e_tr = prepare_data(train, transformer)
    x_val, t_val, e_val = prepare_data(val, transformer)

    return x_tr, t_tr, e_tr, x_val, t_val, e_val

"""
to create a global transformer, calculate these items locally and send globally
+count
+sum
+sum of squares
"""

# Check that it creates the correct datatype (Torch.tensor)
x_train, t_train, e_train, x_val, t_val, e_val = load_datasets(partition_id=0)
print(f"Type of x_train: {type(x_train)}")
print(x_train[:3, :5].round(decimals=2))

"""
def load_datasets(partition_id: int):
    
    partition = fds.load_partition(partition_id)
    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    pytorch_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    def apply_transforms(batch):
        # Instead of passing transforms to CIFAR10(..., transform=transform)
        # we will use this function to dataset.with_transform(apply_transforms)
        # The transforms object is exactly the same
        batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
        return batch

    # Create train/val for each partition and wrap it into DataLoader
    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(
        partition_train_test["train"], batch_size=BATCH_SIZE, shuffle=True
    )
    valloader = DataLoader(partition_train_test["test"], batch_size=BATCH_SIZE)
    testset = fds.load_split("test").with_transform(apply_transforms)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE)
    return trainloader, valloader, testloader
"""





#TODO: dsm_no_fl.py implementation to figure out how to train dsm for a single epoch
# Nothing past this point will run correctly, so I will exit
exit()
class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

"""Let's continue with the usual training and test functions:"""

def train(net, trainloader, epochs: int, verbose=False):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters())
    net.train()
    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        for batch in trainloader:
            images, labels = batch["img"].to(DEVICE), batch["label"].to(DEVICE)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # Metrics
            epoch_loss += loss
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        epoch_loss /= len(trainloader.dataset)
        epoch_acc = correct / total
        if verbose:
            print(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}")


def test(net, testloader):
    """Evaluate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for batch in testloader:
            images, labels = batch["img"].to(DEVICE), batch["label"].to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    loss /= len(testloader.dataset)
    accuracy = correct / total
    return loss, accuracy

"""### Train the model

We now have all the basic building blocks we need: a dataset, a model, a training function, and a test function. Let's put them together to train the model on the dataset of one of our organizations (`partition_id=0`). This simulates the reality of most machine learning projects today: each organization has their own data and trains models only on this internal data:
"""

trainloader, valloader, testloader = load_datasets(partition_id=0)
net = Net().to(DEVICE)

for epoch in range(5):
    train(net, trainloader, 1)
    loss, accuracy = test(net, valloader)
    print(f"Epoch {epoch+1}: validation loss {loss}, accuracy {accuracy}")

loss, accuracy = test(net, testloader)
print(f"Final test set performance:\n\tloss {loss}\n\taccuracy {accuracy}")

"""Training the simple CNN on our CIFAR-10 split for 5 epochs should result in a test set accuracy of about 41%, which is not good, but at the same time, it doesn't really matter for the purposes of this tutorial. The intent was just to show a simple centralized training pipeline that sets the stage for what comes next - federated learning!

## Step 2: Federated Learning with Flower

Step 1 demonstrated a simple centralized training pipeline. All data was in one place (i.e., a single `trainloader` and a single `valloader`). Next, we'll simulate a situation where we have multiple datasets in multiple organizations and where we train a model over these organizations using federated learning.

### Update model parameters

In federated learning, the server sends global model parameters to the client, and the client updates the local model with parameters received from the server. It then trains the model on the local data (which changes the model parameters locally) and sends the updated/changed model parameters back to the server (or, alternatively, it sends just the gradients back to the server, not the full model parameters).

We need two helper functions to update the local model with parameters received from the server and to get the updated model parameters from the local model: `set_parameters` and `get_parameters`. The following two functions do just that for the PyTorch model above.

The details of how this works are not really important here (feel free to consult the PyTorch documentation if you want to learn more). In essence, we use `state_dict` to access PyTorch model parameter tensors. The parameter tensors are then converted to/from a list of NumPy ndarray's (which the Flower `NumPyClient` knows how to serialize/deserialize):
"""

def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

"""### Define the Flower ClientApp

With that out of the way, let's move on to the interesting part. Federated learning systems consist of a server and multiple clients. In Flower, we create a `ServerApp` and a `ClientApp` to run the server-side and client-side code, respectively.

The first step toward creating a `ClientApp` is to implement a subclasses of `flwr.client.Client` or `flwr.client.NumPyClient`. We use `NumPyClient` in this tutorial because it is easier to implement and requires us to write less boilerplate. To implement `NumPyClient`, we create a subclass that implements the three methods `get_parameters`, `fit`, and `evaluate`:

* `get_parameters`: Return the current local model parameters
* `fit`: Receive model parameters from the server, train the model on the local data, and return the updated model parameters to the server
* `evaluate`: Receive model parameters from the server, evaluate the model on the local data, and return the evaluation result to the server

We mentioned that our clients will use the previously defined PyTorch components for model training and evaluation. Let's see a simple Flower client implementation that brings everything together:
"""

class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, valloader):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader

    def get_parameters(self, config):
        return get_parameters(self.net)

    def fit(self, parameters, config):
        set_parameters(self.net, parameters)
        train(self.net, self.trainloader, epochs=1)
        return get_parameters(self.net), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        set_parameters(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}

"""Our class `FlowerClient` defines how local training/evaluation will be performed and allows Flower to call the local training/evaluation through `fit` and `evaluate`. Each instance of `FlowerClient` represents a *single client* in our federated learning system. Federated learning systems have multiple clients (otherwise, there's not much to federate), so each client will be represented by its own instance of `FlowerClient`. If we have, for example, three clients in our workload, then we'd have three instances of `FlowerClient` (one on each of the machines we'd start the client on). Flower calls `FlowerClient.fit` on the respective instance when the server selects a particular client for training (and `FlowerClient.evaluate` for evaluation).

In this notebook, we want to simulate a federated learning system with 10 clients *on a single machine*. This means that the server and all 10 clients will live on a single machine and share resources such as CPU, GPU, and memory. Having 10 clients would mean having 10 instances of `FlowerClient` in memory. Doing this on a single machine can quickly exhaust the available memory resources, even if only a subset of these clients participates in a single round of federated learning.

In addition to the regular capabilities where server and clients run on multiple machines, Flower, therefore, provides special simulation capabilities that create `FlowerClient` instances only when they are actually necessary for training or evaluation. To enable the Flower framework to create clients when necessary, we need to implement a function that creates a `FlowerClient` instance on demand. We typically call this function `client_fn`. Flower calls `client_fn` whenever it needs an instance of one particular client to call `fit` or `evaluate` (those instances are usually discarded after use, so they should not keep any local state). In federated learning experiments using Flower, clients are identified by a partition ID, or `partition-id`. This `partition-id` is used to load different local data partitions for different clients, as can be seen below. The value of `partition-id` is retrieved from the `node_config` dictionary in the `Context` object, which holds the information that persists throughout each training round.

With this, we have the class `FlowerClient` which defines client-side training/evaluation and `client_fn` which allows Flower to create `FlowerClient` instances whenever it needs to call `fit` or `evaluate` on one particular client. Last, but definitely not least, we create an instance of `ClientApp` and pass it the `client_fn`. `ClientApp` is the entrypoint that a running Flower client uses to call your code (as defined in, for example, `FlowerClient.fit`).
"""

def client_fn(context: Context) -> Client:
    """Create a Flower client representing a single organization."""

    # Load model
    net = Net().to(DEVICE)

    # Load data (CIFAR-10)
    # Note: each client gets a different trainloader/valloader, so each client
    # will train and evaluate on their own unique data partition
    # Read the node_config to fetch data partition associated to this node
    partition_id = context.node_config["partition-id"]
    trainloader, valloader, _ = load_datasets(partition_id=partition_id)

    # Create a single Flower client representing a single organization
    # FlowerClient is a subclass of NumPyClient, so we need to call .to_client()
    # to convert it to a subclass of `flwr.client.Client`
    return FlowerClient(net, trainloader, valloader).to_client()


# Create the ClientApp
client = ClientApp(client_fn=client_fn)

"""### Define the Flower ServerApp

On the server side, we need to configure a strategy which encapsulates the federated learning approach/algorithm, for example, *Federated Averaging* (FedAvg). Flower has a number of built-in strategies, but we can also use our own strategy implementations to customize nearly all aspects of the federated learning approach. For this example, we use the built-in `FedAvg` implementation and customize it using a few basic parameters:
"""

# Create FedAvg strategy
strategy = FedAvg(
    fraction_fit=1.0,  # Sample 100% of available clients for training
    fraction_evaluate=0.5,  # Sample 50% of available clients for evaluation
    min_fit_clients=10,  # Never sample less than 10 clients for training
    min_evaluate_clients=5,  # Never sample less than 5 clients for evaluation
    min_available_clients=10,  # Wait until all 10 clients are available
)

"""Similar to `ClientApp`, we create a `ServerApp` using a utility function `server_fn`. In `server_fn`, we pass an instance of `ServerConfig` for defining the number of federated learning rounds (`num_rounds`) and we also pass the previously created `strategy`. The `server_fn` returns a `ServerAppComponents` object containing the settings that define the `ServerApp` behaviour. `ServerApp` is the entrypoint that Flower uses to call all your server-side code (for example, the strategy)."""

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

"""### Run the training

In simulation, we often want to control the amount of resources each client can use. In the next cell, we specify a `backend_config` dictionary with the `client_resources` key (required) for defining the amount of CPU and GPU resources each client can access.
"""

# Specify the resources each of your clients need
# By default, each client will be allocated 1x CPU and 0x GPUs
backend_config = {"client_resources": {"num_cpus": 1, "num_gpus": 0.0}}

# When running on GPU, assign an entire GPU for each client
if DEVICE == "cuda":
    backend_config = {"client_resources": {"num_cpus": 1, "num_gpus": 1.0}}
    # Refer to our Flower framework documentation for more details about Flower simulations
    # and how to set up the `backend_config`

"""The last step is the actual call to `run_simulation` which - you guessed it - runs the simulation. `run_simulation` accepts a number of arguments:
- `server_app` and `client_app`: the previously created `ServerApp` and `ClientApp` objects, respectively
- `num_supernodes`: the number of `SuperNodes` to simulate which equals the number of clients for Flower simulation
- `backend_config`: the resource allocation used in this simulation
"""

# Run simulation
run_simulation(
    server_app=server,
    client_app=client,
    num_supernodes=NUM_CLIENTS, 
    backend_config=backend_config,
)

"""### Behind the scenes

So how does this work? How does Flower execute this simulation?

When we call `run_simulation`, we tell Flower that there are 10 clients (`num_supernodes=10`, where 1 `SuperNode` launches 1 `ClientApp`). Flower then goes ahead an asks the `ServerApp` to issue an instructions to those nodes using the `FedAvg` strategy. `FedAvg` knows that it should select 100% of the available clients (`fraction_fit=1.0`), so it goes ahead and selects 10 random clients (i.e., 100% of 10).

Flower then asks the selected 10 clients to train the model. Each of the 10 `ClientApp` instances receives a message, which causes it to call `client_fn` to create an instance of `FlowerClient`. It then calls `.fit()` on each the `FlowerClient` instances and returns the resulting model parameter updates to the `ServerApp`. When the `ServerApp` receives the model parameter updates from the clients, it hands those updates over to the strategy (*FedAvg*) for aggregation. The strategy aggregates those updates and returns the new global model, which then gets used in the next round of federated learning.

### Where's the accuracy?

You may have noticed that all metrics except for `losses_distributed` are empty. Where did the `{"accuracy": float(accuracy)}` go?

Flower can automatically aggregate losses returned by individual clients, but it cannot do the same for metrics in the generic metrics dictionary (the one with the `accuracy` key). Metrics dictionaries can contain very different kinds of metrics and even key/value pairs that are not metrics at all, so the framework does not (and can not) know how to handle these automatically.

As users, we need to tell the framework how to handle/aggregate these custom metrics, and we do so by passing metric aggregation functions to the strategy. The strategy will then call these functions whenever it receives fit or evaluate metrics from clients. The two possible functions are `fit_metrics_aggregation_fn` and `evaluate_metrics_aggregation_fn`.

Let's create a simple weighted averaging function to aggregate the `accuracy` metric we return from `evaluate`:
"""

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}

def server_fn(context: Context) -> ServerAppComponents:
    """Construct components that set the ServerApp behaviour.

    You can use settings in `context.run_config` to parameterize the
    construction of all elements (e.g the strategy or the number of rounds)
    wrapped in the returned ServerAppComponents object.
    """

    # Create FedAvg strategy
    strategy = FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=0.5,
        min_fit_clients=10,
        min_evaluate_clients=5,
        min_available_clients=10,
        evaluate_metrics_aggregation_fn=weighted_average,  # <-- pass the metric aggregation function
    )

    # Configure the server for 5 rounds of training
    config = ServerConfig(num_rounds=5)

    return ServerAppComponents(strategy=strategy, config=config)


# Create a new server instance with the updated FedAvg strategy
server = ServerApp(server_fn=server_fn)

# Run simulation
run_simulation(
    server_app=server,
    client_app=client,
    num_supernodes=NUM_CLIENTS,
    backend_config=backend_config,
)

"""We now have a full system that performs federated training and federated evaluation. It uses the `weighted_average` function to aggregate custom evaluation metrics and calculates a single `accuracy` metric across all clients on the server side.

The other two categories of metrics (`losses_centralized` and `metrics_centralized`) are still empty because they only apply when centralized evaluation is being used. Part two of the Flower tutorial will cover centralized evaluation.

## Final remarks

Congratulations, you just trained a convolutional neural network, federated over 10 clients! With that, you understand the basics of federated learning with Flower. The same approach you've seen can be used with other machine learning frameworks (not just PyTorch) and tasks (not just CIFAR-10 images classification), for example NLP with Hugging Face Transformers or speech with SpeechBrain.

In the next notebook, we're going to cover some more advanced concepts. Want to customize your strategy? Initialize parameters on the server side? Or evaluate the aggregated model on the server side? We'll cover all this and more in the next tutorial.

## Next steps

Before you continue, make sure to join the Flower community on Flower Discuss ([Join Flower Discuss](https://discuss.flower.ai)) and on Slack ([Join Slack](https://flower.ai/join-slack/)).

There's a dedicated `#questions` channel if you need help, but we'd also love to hear who you are in `#introductions`!

The [Flower Federated Learning Tutorial - Part 2](https://flower.ai/docs/framework/tutorial-use-a-federated-learning-strategy-pytorch.html) goes into more depth about strategies and all the advanced things you can build with them.
"""