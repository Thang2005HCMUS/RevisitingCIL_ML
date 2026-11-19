a = """
cifar224_5_5_vit-b_adapter.json  imageneta_10_10_vit-b_adapter.json  objectnet_10_10_vit-b_adapter.json  vtab_10_10_vit-b_adapter.json
cub_10_10_vit-b_adapter.json     imagenetr_5_5_vit-b_adapter.json    omni_30_30_vit-b_adapter.json
"""
b = a.split()
for i in b:
    print(f"!python main.py --config ./exps/aper_adapter/{i} --wandb")