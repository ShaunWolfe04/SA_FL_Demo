# SA_FL_Demo
Federated Learning in survival analysis environments

to run: pip install -r requirements.txt
(conda env coming soon?)

### flower_dsm.py: 
Deep Survival Machines integrated into Flower \
command to run: \
`TQDM_DISABLE=1 python3 flower_dsm.py` (less noisy output)
* Status
    - Complete, but has worse accuracy than non-FL counterpart
* Changelog
    * 06/27/25
        * Added rough version of early stopping
        * Added support for model extraction after simulation
        * Added post-simulation evalutation
    * 06/25/25
        * Fixed noisy output with deprecation warnings and tqdm
        * Refactored DSMModel to inherit from DSMBase
* Future improvements:
    * Refactor to allow for clustering
    * Attempt to improve accuracy

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

