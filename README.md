# quality-diversity-pretraining
Code for our ICCV paper paper - Quality Diversity for Visual Pre-training

## Requirements
This code base has been tested with the following package versions:

```
python=3.8.13
torch=1.13.0
torchvision=0.14.0
PIL=7.1.2
numpy=1.22.3
scipy=1.7.3
tqdm=4.31.1
sklearn=1.2.1
wandb=0.13.4
tllib=0.4
```

For pretraining download [ImageNet](https://www.image-net.org) and generate ImageNet-100 using this [repository](https://github.com/danielchyeh/ImageNet-100-Pytorch). 

Make a folder named ```TestDatasets``` to download and process downstream datasets. Below is the outline of expected file structure. 

```
imagenet1k/
imagenet-100/
amortized-invariance-learning-ssl/
    saved_models/
    ...
TestDatasets/
    CIFAR10/
    CIFAR100/
    300w/
    ...
```

## Pre-training

To run our QD supervised pre-training with ImageNet1k dataset, run the following command in ```supervised/```

```
python train.py --multiprocessing-distributed --rank 0 --world-size 1 --dist-url "tcp://localhost:10001" --train_data imagenet1k --data <path to data> --num_augs 6 --batch-size 1024 
--lr 0.5 --lr-scheduler cosineannealinglr --lr-warmup-epochs 5 --lr-warmup-method linear --lr-warmup-decay 0.01 --wd 2e-5 --mixup --cutmix --model-ema --output_dir <dir to save models>
```


## Downstream training

We evaluate on several downstream datasets including [CIFAR10](https://pytorch.org/vision/stable/datasets.html), [CIFAR100](https://pytorch.org/vision/stable/datasets.html), [Caltech101](http://www.vision.caltech.edu/Image_Datasets/Caltech101/), [DTD](https://www.robots.ox.ac.uk/~vgg/data/dtd/), [Oxford-Flowers](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) [300w](https://ibug.doc.ic.ac.uk/resources/300-W/), [Leeds Sports Pose](https://dbcollection.readthedocs.io/en/latest/datasets/leeds_sports_pose_extended.html), and [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html). We download these datasets in ```../TestDatasets/```. Training and test splits have been adopted from the [Transfer Learning Library](https://github.com/thuml/Transfer-Learning-Library/tree/master)

To run downstream experiments for QD models for CIFAR10, CIFAR100, CalTech101, DTD, OxfordFlowers102, StanfordCars, Aircraft, Animal Pose, MPII, ALOI, and Causal 3D run 
```
python main_linear.py -a <arch> --test_dataset <dataset name> --gpu 0 --pretrained saved_models/<name of checkpoint> 
```
```dataset_name``` for each dataset can be found in ```downstream_utils.py```

For evaluation on 300w, Leeds Sports Pose and CelebA, run
```
python main_linear_nn.py -a <arch> --test_dataset <dataset name> --gpu 0 --pretrained saved_models/<name of checkpoint> 
```

For few-shot evaluation run, 
```
python few_shot.py -a <arch> --test_dataset <dataset name> --gpu 0 --pretrained saved_models/<name of checkpoint> 
```
```
```dataset_name``` for each dataset can be found in ```few_shot.py```

If you find our work helpful, please cite our paper
```
@InProceedings{Chavhan_2023_ICCV,
    author    = {Chavhan, Ruchika and Gouk, Henry and Li, Da and Hospedales, Timothy},
    title     = {Quality Diversity for Visual Pre-Training},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {5384-5394}
}
```

