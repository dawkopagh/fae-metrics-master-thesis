# Trained Weights — ISIC 2017 (3-class skin lesion classification)

## Source

- **Trained on**: Google Colab, Tesla T4 GPU
- **Date**: 2026-04-22
- **Drive path**: `/content/drive/MyDrive/thesis/weights/`
- **Git commit used for training**: `a554b99` (feat: ISIC 2017 download + training pipeline + Colab notebook)

## Training Configuration

| Parameter             | Value                                        |
|-----------------------|----------------------------------------------|
| Epochs                | 25                                           |
| Seed                  | 42                                           |
| Optimizer             | SGD + momentum                               |
| Scheduler             | StepLR                                       |
| Loss                  | Class-weighted CrossEntropyLoss              |
| Device                | CUDA (Tesla T4)                              |
| Dataset               | Full ISIC 2017 (2000 train / 150 val / 600 test) |
| Classes               | melanoma, nevus, seborrheic_keratosis        |

## Validation Accuracy

| Model          | Best Val Accuracy |
|----------------|-------------------|
| ResNet-18      | 0.7533            |
| SqueezeNet 1.1 | 0.7400            |

## SHA-256 Checksums

```
593dcb844b8359550e3d84667475bbd845f45a6cb388356480c0a55cb5173430  resnet18_isic2017.pth
8bbb43bbba4ee81e58e354295420bea33e55cfa2be11c31d82dce272d03b092d  squeezenet_isic2017.pth
```

## Legacy Weights

Old 75-image weights preserved at `legacy/weights/` for reference:
- `resnet_skin_75img.pth` — ResNet-18 trained on 75-image subset
- `squeezenet_skin_75img.pth` — SqueezeNet trained on 75-image subset

**Note**: The old weights (`resnet_skin.pth`, `squeezenet_skin.pth`) in the
project root are the same 75-image weights and should be moved to
`legacy/weights/` via:
```
git mv resnet_skin.pth   legacy/weights/resnet_skin_75img.pth
git mv squeezenet_skin.pth legacy/weights/squeezenet_skin_75img.pth
```
