## GRAND minimal codes

Main function is `run_GNN.py`

Inside the main, `DATA_DIR` needs to be checked.

Environment is in `/ram/USERS/ziquanw/softwares/miniconda3/envs/GRAND`

### To run

To run `python run_GNN.py`, you need to set `--dataset` for different experiments.

 - For graph classification, `python run_GNN.py --dataset "Tau_classificationcus_cls-2"`, this means the data under the folder `Tau_classification2` will be used. Note, `cus_cls-` in the middle is meaningless, it is a string for code to know which experiment is running.

 - For signal prediction, `python run_GNN.py --dataset "customTau/Tau_prediction"`, this means the data under the folder `Tau/Tau_prediction` will be used. Note, `custom` in the beginning is meaningless, it is a string for code to know which experiment is running.

### Example

 - For graph classification, `run_Tau_cls.sh`
 - For signal prediction, `run_Tau.sh`
