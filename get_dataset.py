from auton_survival.datasets import load_dataset
from auton_survival.preprocessing import Preprocessor
from flwr_datasets import FederatedDataset
from sklearn.model_selection import train_test_split
import pandas as pd
from flwr_datasets.partitioner import DirichletPartitioner
from pycox.datasets import metabric
from flwr_datasets.partitioner.partitioner import Partitioner
from datasets import Dataset
import os


def get_dataset(dataset, split, NUM_CLIENTS):
    if dataset == "SUPPORT":
        return get_support(split, NUM_CLIENTS)
    if dataset == "METABRIC":
        print("Warning: this setup of DSM is unable to train on the METABRIC dataset")
        return get_metabric(split, NUM_CLIENTS) # Not implemented yet
    if dataset == "SEER":
        print("Warning: this setup of DSM is unable to train on the SEER dataset")
        return get_seer(split, NUM_CLIENTS)
    if dataset == "FRAMINGHAM":
        return get_fh(split, NUM_CLIENTS)
    if dataset == "PBC":
        return get_pbc(split, NUM_CLIENTS) 
    else:
        raise ValueError(f"Dataset unknown: {dataset}")
        


def get_support(split, NUM_CLIENTS):
    
    #load the data
    outcomes, features = load_dataset(dataset='SUPPORT')
    
    #preprocess the data
    cat_feats = ['sex', 'dzgroup', 'dzclass', 'income', 'race', 'ca']
    num_feats = ['age', 'num.co', 'meanbp', 'wblc', 'hrt', 'resp', 
                'temp', 'pafi', 'alb', 'bili', 'crea', 'sod', 'ph', 
                'glucose', 'bun', 'urine', 'adlp', 'adls']

    preprocessor = Preprocessor(cat_feat_strat='ignore', num_feat_strat= 'mean') 
    transformer = preprocessor.fit(features, cat_feats=cat_feats, num_feats=num_feats,
                                    one_hot=True, fill_value=-1)
    features_upd = transformer.transform(features)

    # re-add data to split on for dirichlet
    features_upd["_partition"]=features["dzgroup"]
    
    #create test-train split
    x_tr, x_te, y_tr, y_te = train_test_split(features_upd, outcomes, test_size=0.2, random_state=1)

    # drop partition for test (dropping for train is done later)
    x_te=x_te.drop(["_partition"], axis=1)
    
    #convert to file so that FederatedDataset can be called
    combined_df = pd.concat([x_tr, y_tr], axis=1)
    temp_csv_path = "support_dataset_temp.csv"
    combined_df.to_csv(temp_csv_path, index=False)

    # splitting strategy
    if split == "dirichlet":
        fds = apply_dirichlet(temp_csv_path, NUM_CLIENTS)
    elif split == "iid": 
        fds = FederatedDataset(dataset="csv", 
                            partitioners={"train": NUM_CLIENTS}, 
                            data_files=temp_csv_path)
    else:
        raise Exception("location not implemented for METABRIC")
    return x_tr, x_te, y_tr, y_te, fds

def get_metabric(split, NUM_CLIENTS):
    df = metabric.read_df()
    df.rename(columns={'duration': 'time'}, inplace=True)

    #split into x and y
    outcome_cols = ['time', 'event']
    outcomes = df[outcome_cols]
    features = df.drop(columns=outcome_cols)

    time_bins = pd.qcut(outcomes['time'], q=4, labels=False, duplicates='drop')

    # Create partition split by using times
    features['_partition'] = time_bins
    x_tr, x_te, y_tr, y_te = train_test_split(features, outcomes, test_size=0.2, random_state=1)

    # drop partition for test (dropping for train is done later)
    x_te=x_te.drop(["_partition"], axis=1)

    combined_df = pd.concat([x_tr, y_tr], axis=1)
    temp_csv_path="metabric.csv"
    combined_df.to_csv(temp_csv_path, index=False)

    if split == "dirichlet":
        fds = apply_dirichlet(temp_csv_path, NUM_CLIENTS)
    elif split == "iid": 
        fds = FederatedDataset(dataset="csv",
                            partitioners={"train": NUM_CLIENTS}, 
                            data_files=temp_csv_path
        )
    else:
        raise Exception("location not implemented for METABRIC")
    return x_tr.astype('float64'), x_te.astype('float64'), y_tr, y_te, fds


def get_seer(split, NUM_CLIENTS):

    if split == "dirichlet":
        raise Exception("dirichlet has not been implemented for SEER")
        # but you can easily do it yourself
    df = pd.read_csv('SEER.csv')
    df = df.sample(frac=0.1, random_state=42)
    df.rename(columns={'d.time': 'time'}, inplace=True)
    df.rename(columns={'death': 'event'}, inplace=True)

    outcome_cols = ['time', 'event']
    outcomes = df[outcome_cols]
    features = df.drop(columns=outcome_cols)


    cols_to_drop = ['Patient ID', 'HisTICD03', 'State/City']
    features = features.drop(columns=cols_to_drop)
    
    # Here we are dropping Patient ID (irrelevant) and HisTICD03 (too many features)
    cat_feats = ['Sex', 'R/E', 'ROriginCd', 'Marital', 'DiagConf', 'PSite', 'HisCdGrp',
     'Grade', 'Laterality', 'TNMalTumor', 'TNBenTumor', 'RadCd', 'ChemCd', 'RepSrc']
    num_feats = ['Year', 'Age']

    # Preprocess
    preprocessor = Preprocessor(cat_feat_strat='ignore', num_feat_strat= 'mean') 
    transformer = preprocessor.fit(features, cat_feats=cat_feats, num_feats=num_feats,
                                    one_hot=True, fill_value=-1)
    features_upd = transformer.transform(features)
    print(f"features_upd shape: {features_upd.shape}")
    features_upd["_partition"] = df['State/City']

    x_tr, x_te, y_tr, y_te = train_test_split(features_upd, outcomes, test_size=0.2, random_state=1)

    # drop partition for test (dropping for train is done later)
    x_te=x_te.drop(["_partition"], axis=1)
    
    #convert to file so that FederatedDataset can be called
    combined_df = pd.concat([x_tr, y_tr], axis=1)
    temp_csv_path = "seer_dataset_temp.csv"
    combined_df.to_csv(temp_csv_path, index=False)

    if split == "location":
        partition_col = "_partition"
        partition_dir = "SEER_partitions"
        os.makedirs(partition_dir, exist_ok=True) # Create the directory

        grouped = combined_df.groupby(partition_col)
        unique_categories = []
        i = 0
        for group_name, group_df in grouped:
            # Sanitize group_name for use in a filename if necessary
            safe_group_name = str(group_name).replace('/', '_')
            partition_filename = os.path.join(partition_dir, f"partition_{i}.csv")
            group_df.to_csv(partition_filename, index=False)
            i+=1
        partitioner = SingleCategoryPartitioner("_partition")
        fds = FederatedDataset(dataset="csv",
                            partitioners = {"train": partitioner},
                            data_files = temp_csv_path)
    else:
        fds = FederatedDataset(dataset="csv", 
                            partitioners = {"train": NUM_CLIENTS}, 
                            data_files = temp_csv_path)
    return x_tr, x_te, y_tr, y_te, fds


def get_fh(split, NUM_CLIENTS):

    x, t, e = load_dataset(dataset="FRAMINGHAM")
    features = pd.DataFrame(x)
    outcomes = pd.DataFrame({'time': t, 'event':e })
    time_bins = pd.qcut(outcomes['time'], q=4, labels=False, duplicates='drop')

    # Create partition split by using times
    features['_partition'] = time_bins
    x_tr, x_te, y_tr, y_te = train_test_split(features, outcomes, test_size=0.2, random_state=1)

    # drop partition for test (dropping for train is done later)
    x_te=x_te.drop(["_partition"], axis=1)

    combined_df = pd.concat([x_tr, y_tr], axis=1)
    temp_csv_path="framingham.csv"
    combined_df.to_csv(temp_csv_path, index=False)

    if split == "dirichlet":
        fds = apply_dirichlet(temp_csv_path, NUM_CLIENTS)
    elif split == "iid": 
        fds = FederatedDataset(dataset="csv",
                            partitioners={"train": NUM_CLIENTS}, 
                            data_files=temp_csv_path
        )
    else:
        raise Exception("location not implemented for FRAMINGHAM")
    return x_tr.astype('float64'), x_te.astype('float64'), y_tr, y_te, fds
    
def get_pbc(split, NUM_CLIENTS):

    x, t, e = load_dataset(dataset="PBC")
    features = pd.DataFrame(x)
    outcomes = pd.DataFrame({'time': t, 'event':e })
    time_bins = pd.qcut(outcomes['time'], q=4, labels=False, duplicates='drop')

    # Create partition split by using times
    features['_partition'] = time_bins
    x_tr, x_te, y_tr, y_te = train_test_split(features, outcomes, test_size=0.2, random_state=1)

    # drop partition for test (dropping for train is done later)
    x_te=x_te.drop(["_partition"], axis=1)

    combined_df = pd.concat([x_tr, y_tr], axis=1)
    temp_csv_path="pbc.csv"
    combined_df.to_csv(temp_csv_path, index=False)

    if split == "dirichlet":
        fds = apply_dirichlet(temp_csv_path, NUM_CLIENTS)
    elif split == "iid": 
        fds = FederatedDataset(dataset="csv",
                            partitioners={"train": NUM_CLIENTS}, 
                            data_files=temp_csv_path
        )
    else:
        raise Exception("location not implemented for PBC")
    return x_tr.astype('float64'), x_te.astype('float64'), y_tr, y_te, fds

def apply_dirichlet(temp_csv_path, NUM_CLIENTS):

    # SUPPORT is partitioned on "dzgroup"
    # METABRIC is partitioned on time
    # SEER is not tested with dirichlet
    partitioner = DirichletPartitioner(
        num_partitions=NUM_CLIENTS,
        partition_by="_partition",
        alpha=0.5,
        seed=42 
    )

    fds = FederatedDataset(
        dataset="csv",
        partitioners={"train": partitioner}, 
        data_files=temp_csv_path
    )
    return fds


class SingleCategoryPartitioner(Partitioner):
    """Partitions a dataset so that each partition contains all data
    for exactly one unique category from a specified column.

    Args:
        partition_by (str): The name of the column to partition by.
    """
    def __init__(self, partition_by: str) -> None:
        super().__init__()
        self.partition_by = partition_by
        self._unique_categories = None

    def load_partition(self, partition_id: int) -> Dataset:
        """Load a single partition of the dataset.

        Args:
            partition_id (int): The integer ID of the partition to load,
                corresponding to the index of the unique category.

        Returns:
            Dataset: The loaded partition.
        """
        # Get the full dataset from the parent class
        full_dataset = self.dataset

        # If we haven't identified the unique categories yet, do it now.
        if self._unique_categories is None:
            self._unique_categories = sorted(full_dataset.unique(self.partition_by))
            print(f"Found {len(self._unique_categories)} unique categories to partition by: {self._unique_categories}")

        # Check if the requested partition_id is valid
        if partition_id >= len(self._unique_categories):
            raise ValueError(
                f"Partition ID {partition_id} is out of bounds. "
                f"There are only {len(self._unique_categories)} unique categories."
            )

        # Retrieve dataset
        filename = f"SEER_partitions/partition_{partition_id}.csv"
        print(f"retrieving data from {filename}")
        partition_dataset = pd.read_csv(filename)

        return Dataset.from_pandas(partition_dataset)

    @property
    def num_partitions(self) -> int:
        """Returns the total number of partitions."""
        if self._unique_categories is None:
            self._unique_categories = sorted(self.dataset.unique(self.partition_by))
        return len(self._unique_categories)
