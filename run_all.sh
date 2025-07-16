#!/bin/bash

# This script runs all federated learning experiments sequentially.
# It creates directories for logs and results to keep the project organized.

# --- Record the start time ---
start_time=$(date +%s)

echo "--- Starting Full Experimental Run ---"
echo "Start Time: $(date)"

# Create directories for logs and results if they don't exist
mkdir -p logs
mkdir -p results
mkdir -p plots

# Clear the old results file to start fresh for this run
> results/results.txt

# --- SUPPORT Dataset Experiments ---
echo "--- $(date) Running SUPPORT | IID | FedAvg ---"
python3 flower_dsm.py --dataset SUPPORT --split iid --strategy avg > logs/support_iid_avg.log 2>&1

echo "--- $(date) Running SUPPORT | IID | FedProx 0.01 ---"
python3 flower_dsm.py --dataset SUPPORT --split iid --strategy prox --prox_mu 0.01 > logs/support_iid_prox001.log 2>&1

echo "--- $(date) Running SUPPORT | IID | FedProx 0.1 ---"
python3 flower_dsm.py --dataset SUPPORT --split iid --strategy prox --prox_mu 0.1 > logs/support_iid_prox01.log 2>&1

echo "--- $(date) Running SUPPORT | IID | FedProx 1 ---"
python3 flower_dsm.py --dataset SUPPORT --split iid --strategy prox --prox_mu 1 > logs/support_iid_prox1.log 2>&1

echo "--- $(date) Running SUPPORT | dirichlet | FedAvg ---"
python3 flower_dsm.py --dataset SUPPORT --split dirichlet --strategy avg > logs/support_dirichlet_avg.log 2>&1

echo "--- $(date) Running SUPPORT | dirichlet | FedProx 0.01 ---"
python3 flower_dsm.py --dataset SUPPORT --split dirichlet --strategy prox --prox_mu 0.01 > logs/support_dirichlet_prox001.log 2>&1

echo "--- $(date) Running SUPPORT | dirichlet | FedProx 0.1 ---"
python3 flower_dsm.py --dataset SUPPORT --split dirichlet --strategy prox --prox_mu 0.1 > logs/support_dirichlet_prox01.log 2>&1

echo "--- $(date) Running SUPPORT | dirichlet | FedProx 1 ---"
python3 flower_dsm.py --dataset SUPPORT --split dirichlet --strategy prox --prox_mu 1 > logs/support_dirichlet_prox1.log 2>&1


# --- METABRIC Dataset Experiments ---
echo "--- $(date) Running METABRIC | IID | FedAvg ---"
python3 flower_dsm.py --dataset METABRIC --split iid --strategy avg > logs/metabric_iid_avg.log 2>&1

echo "--- $(date) Running METABRIC | IID | FedProx 0.01 ---"
python3 flower_dsm.py --dataset METABRIC --split iid --strategy prox --prox_mu 0.01 > logs/metabric_iid_prox001.log 2>&1

echo "--- $(date) Running METABRIC | IID | FedProx 0.1 ---"
python3 flower_dsm.py --dataset METABRIC --split iid --strategy prox --prox_mu 0.1 > logs/metabric_iid_prox01.log 2>&1

echo "--- $(date) Running METABRIC | IID | FedProx 1 ---"
python3 flower_dsm.py --dataset METABRIC --split iid --strategy prox --prox_mu 1 > logs/metabric_iid_prox1.log 2>&1

echo "--- $(date) Running METABRIC | dirichlet | FedAvg ---"
python3 flower_dsm.py --dataset METABRIC --split dirichlet --strategy avg > logs/metabric_dirichlet_avg.log 2>&1

echo "--- $(date) Running METABRIC | dirichlet | FedProx 0.01 ---"
python3 flower_dsm.py --dataset METABRIC --split dirichlet --strategy prox --prox_mu 0.01 > logs/metabric_dirichlet_prox001.log 2>&1

echo "--- $(date) Running METABRIC | dirichlet | FedProx 0.1 ---"
python3 flower_dsm.py --dataset METABRIC --split dirichlet --strategy prox --prox_mu 0.1 > logs/metabric_dirichlet_prox01.log 2>&1

echo "--- $(date) Running METABRIC | dirichlet | FedProx 1 ---"
python3 flower_dsm.py --dataset METABRIC --split dirichlet --strategy prox --prox_mu 1 > logs/metabric_dirichlet_prox1.log 2>&1


# --- SEER Dataset Experiments ---
echo "--- $(date) Running SEER | IID | FedAvg ---"
python3 flower_dsm.py --dataset SEER --split iid --strategy avg > logs/seer_iid_avg.log 2>&1

echo "--- $(date) Running SEER | IID | FedProx 0.1 ---"
python3 flower_dsm.py --dataset SEER --split iid --strategy prox --prox_mu 0.1 > logs/seer_iid_prox01.log 2>&1

echo "--- $(date) Running SEER | location | FedAvg ---"
python3 flower_dsm.py --dataset SEER --split location --strategy avg > logs/seer_location_avg.log 2>&1

echo "--- $(date) Running SEER | location | FedProx 0.1 ---"
python3 flower_dsm.py --dataset SEER --split location --strategy prox --prox_mu 0.1 > logs/seer_location_prox01.log 2>&1

# ... and so on for all your other experiment combinations ...


# --- Calculate and display the total runtime ---
end_time=$(date +%s)
duration=$((end_time - start_time))
minutes=$((duration / 60))
seconds=$((duration % 60))

echo "----------------------------------------"
echo "--- All Experiments Complete ---"
echo "End Time: $(date)"
echo "Total Runtime: ${minutes} minutes and ${seconds} seconds."
echo "----------------------------------------"