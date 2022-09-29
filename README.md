# Anyang_AttributeRecognition

__Installation__
---
```
git clone https://github.com/jicheol93/Anyang_ar.git
cd anyang_ar
conda env create -f requirements.yml
```

__Prepare Anyang Attribute Recognition dataset__
---
```
Anyang_ar
└── Anyang_data
    ├── Attribute_labeling.xlsx
    ├── train
    └── test

cd anyang_ar
(change data path from prepare_Anyang_xxx.py)
python prepare_Anyang_train.py --h
python prepare_Anyang_train.py --h
(--h option means filtering data lower than the height of 150 pixels.)
```
__Train__
---
```
python train_att_backbone_from_AR_Anyang.py --AR --name AR_Anyang_high_lr001_b128 --lr 0.001 --batchsize 128 --h --data_dir "/home/jicheol/Anyang_ar/Anyang_data/pytorch/"
```

__Inference__
---
```
 python test_att_backbone_from_AR_Anyang.py --AR --name AR_Anyang_high_lr001_b128 --batchsize 128 --h --data_dir "/home/jicheol/Anyang_ar/Anyang_data/pytorch/" --which_epoch "last"
```
__Visualization__
---
```
 python test_att_backbone_from_AR_Anyang.py --AR --name AR_Anyang_high_lr001_b128 --batchsize 128 --h --vis --data_dir "/home/jicheol/Anyang_ar/Anyang_data/pytorch/" --which_epoch "last"
<<<<<<< HEAD
 
 Anyang_ar
└── vis_image
(visualization results are saved in vis_image)

=======

 Anyang_ar
└── vis_image
(visualization results are saved in vis_image)
![image](https://user-images.githubusercontent.com/39580015/187052365-3194588d-a5a9-420b-8e88-bd4ea831d48b.png)
>>>>>>> 8246898613ce391d450ef91256705c2173fbe22c
```
