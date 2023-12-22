# cus_cls- is a meaningless string to let code using the correct custom dataset
python run_GNN.py --dataset "Tau_classificationcus_cls-2" --time 2.0 --gpu 1 > Tau_out_cls_GTV1.txt
python run_GNN.py --dataset "Tau_classificationcus_cls-_CN_AD" --time 2.0 --gpu 1 > Tau_out_GTV1.txt

python run_GNN.py --dataset "Tau_classificationcus_cls-_CN_AD" --time 2.0 --gpu 1 > Tau_out_OASIS.txt