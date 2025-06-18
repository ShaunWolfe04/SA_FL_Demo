# SA_FL_Demo
Federated Learning in survival analysis environments

to run: pip install -r requirements.txt
(conda env coming soon?)

flower_template.py: a template for implementing survival analysis in Flower
- Partition dataset: complete
- Transform dataset: complete
- Training loop: in progress
- client / server functions: needs tweaking

dsm_no_fl.py: implementing dsm without using the suggested "SurvivalModel" approach
Need to do this since the functions in SurvivalModel are monolithic

