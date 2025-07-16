# SA_FL_Demo
Federated Learning in survival analysis environments

to run: pip install -r requirements.txt
(conda env coming soon?)

### flower_dsm.py: 
Deep Survival Machines integrated into Flower \
Uses FedAvg \
command to run: \
`TQDM_DISABLE=1 python3 flower_dsm.py <strategy>` (less noisy output) \
available strategy keywords: avg, prox. Default: avg \
currently creates a lot of files, so be warned
* Status
    - Complete with support for multiple data splits and multiple datasets
* Changelog
    * 07/16/25
        * Added support to pick SUPPORT, METABRIC, or SEER as your dataset
        * Can pick between iid and dirichlet distributions for SUPPORT and METABRIC
        * Can pick between iid distribution and a location-based distribution for SEER
        * (SEER data not provided)
    * 07/02/25
        * Allow for multiple strategies to be used
        * Currently supported: FedAvg and FedProx
    * 07/01/25
        * Small update to improve performance
    * 06/27/25
        * Added rough version of early stopping
        * Added support for model extraction after simulation
        * Added post-simulation evalutation
        * Speed up runtime by around half a minute
    * 06/25/25
        * Fixed noisy output with deprecation warnings and tqdm
        * Refactored DSMModel to inherit from DSMBase
    * 07/02/25
* Future improvements:
    * Refactor to allow for clustering
    * Attempt to improve accuracy
    * Make it so that it does not create 50 files in your wd


### get_dataset.py
Extra framework for dataset and distribution selection for flower_dsm.py
* Status
    * Supports SUPPORT, METABRIC, and SEER (SEER data not provided)
* Changelog
    * 07/16/2025
        * init
### strategy.py
Extra framework for strategy selection in flower_dsm.py
* Status
    - Has support for FedAvg and FedProx, and also saves models for accuracy calculation by default
* Changelog
    * 07/02/25
        * init


### flower_template.py: 
a template for implementing survival analysis in Flower. \
Currently just a broken version of flower_dsm.py without train/server/client initialized
- Partition dataset: complete
- Transform dataset: complete
- Training loop: Not implemented
- client / server functions: Not implemented
- Needs major updating, will not run in current state

### dsm_no_fl.py: 
implementing dsm without using the suggested "SurvivalModel" approach \
Need to do this since the functions in SurvivalModel are monolithic
* Status
    * Complete with a couple small improvements needed
* Changelog
    * 06/25/25
        * Refactored DSMModel to inherit from DSMBase
        * Added accuracy metrics
        * Fixed bug with seemingly exploding gradients

