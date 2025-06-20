# SA_FL_Demo
Federated Learning in survival analysis environments

to run: pip install -r requirements.txt
(conda env coming soon?)

### flower_dsm.py: 
Deep Survival Machines integrated into Flower
Works, but something is incorrect in the training loop, does not train correctly
- Partition dataset: done
- Transform dataset: done globally
- Training loop: complete with a small bug
- client / server loop: complete but very noisy

### flower_template.py: 
a template for implementing survival analysis in Flower. 
Currently just a broken version of flower_dsm.py without train/server/client initialized
- Partition dataset: complete
- Transform dataset: complete
- Training loop: Not implemented
- client / server functions: Not implemented

### dsm_no_fl.py: 
implementing dsm without using the suggested "SurvivalModel" approach
Need to do this since the functions in SurvivalModel are monolithic
* Mostly complete, but has a bug in the training loop. Subject to exploding gradients.

