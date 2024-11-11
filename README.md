# Class token Knowledge Distillation

## Usage

### Requirements

```
pytorch==1.8.0
timm==0.5.4
```

### Data preparation

Download and extract ImageNet train and val images from http://image-net.org/.
The directory structure is:

```
│path/to/imagenet/
├──train/
│  ├── n01440764
│  │   ├── n01440764_10026.JPEG
│  │   ├── n01440764_10027.JPEG
│  │   ├── ......
│  ├── ......
├──val/
│  ├── n01440764
│  │   ├── ILSVRC2012_val_00000293.JPEG
│  │   ├── ILSVRC2012_val_00002138.JPEG
│  │   ├── ......
│  ├── ......
```

### Training on ImageNet-1K

To train a DeiT-Tiny student with a DeiT-B teacher, run:

```shell
python -m torch.distributed.launch --nproc_per_node=4 main.py --distributed --output_dir <output-dir> --data-path <dataset-dir> --teacher-path <path-of-teacher-checkpoint> --model deit_tiny_patch16_224 --teacher-model deit_base_patch16_224 --distillation-type soft --distillation-alpha 1 --distillation-beta 1 --w-cls 1.0 --w-atn 2.0 --w-sample 0.1 --w-patch 4 --w-rand 0.2 --K 192 --s-id 0 1 2 3 8 9 10 11 --t-id 0 1 2 3 8 9 10 11 --drop-path 0 --batch-size 128 --seed 0 --manifold
```

To train a DeiT-Tiny student with a TNT-S teacher, run:

```shell
python -m torch.distributed.launch --nproc_per_node=4 main.py --distributed --output_dir <output-dir> --data-path <dataset-dir> --teacher-path <path-of-teacher-checkpoint> --model deit_tiny_patch16_224 --teacher-model tnt_s_patch16_224 --distillation-type soft --distillation-alpha 0.5 --distillation-beta 1 --w-sample 0.1 --w-patch 4 --w-rand 0.2 --K 192 --s-id 0 1 2 3 8 9 10 11 --t-id 0 1 2 3 8 9 10 11 --drop-path 0 --batch-size 128 --seed 0
```

### Evaluation on ImageNet-1K

python main.py --eval --resume /path/to/trained-ckpt --data-path /path/to/imagenet
