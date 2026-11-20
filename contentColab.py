a = """
cifar224_5_5_vit-b_vpt-deep.json     objectnet_10_10_vit-b_vpt-deep.json
cub_10_10_vit-b_vpt-deep.json        omni_30_30_vit-b_vpt-deep.json
imageneta_10_10_vit-b_vpt-deep.json  vtab_10_10_vit-b_vpt-deep.json
imagenetr_5_5_vit-b_vpt-deep.json
"""
b = a.split()
l = 1
for i in b:
    l+=1
    print(f"!nohup python main.py --config ./exps/aper_vpt-deep/{i} --wandb > log_{i}.txt 2>&1 &")