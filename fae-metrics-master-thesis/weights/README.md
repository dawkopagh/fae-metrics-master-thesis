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

| Filename | SHA-256 | Notes |
|---|---|---|
| `resnet_skin_75img.pth` | `49c65a38ec9449edfc2b04bdbbf5ceae29499434e91fbf26e84620a04abeacc1` | ResNet-18 trained on 75-image subset (pre-thesis scope) |
| `squeezenet_skin_75img.pth` | `d0f07aecabcd78b2b4bd8e7c13a8f9eac626820e216ae3db45d4f6633997d9ef` | SqueezeNet trained on 75-image subset (pre-thesis scope) |

These files were previously at the project root as `resnet_skin.pth` and
`squeezenet_skin.pth` and moved to `legacy/weights/` on 2026-05-17.
