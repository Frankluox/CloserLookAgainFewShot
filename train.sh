#test
# python write_yaml_test.py
# python main.py --cfg configs/evaluation/finetune_res12_CE.yaml --tag test

# training CE model
# python write_yaml_CE.py
# python main.py --cfg configs/CE/miniImageNet_res12.yaml --tag main

# training PN model
# python write_yaml_PN.py
# python main.py --cfg configs/PN/miniImageNet_res12.yaml --tag main

# searching for hyperparameters for finetune.
python write_yaml_search.py
# python search_hyperparameter.py --cfg configs/search/finetune_res12_CE.yaml
