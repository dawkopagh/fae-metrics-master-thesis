"""
Configuration Parameters and Settings

This code snippet defines a set of configuration parameters and settings for a deep learning image analysis pipeline.
It includes specifications for the model, training, evaluation, data paths, class definitions, and explanation methods.

Usage:
    - Modify the values of the parameters and settings according to your specific use case.
    - These parameters are used to configure the image analysis process and explainability methods.

Dependencies:
    - torch: PyTorch library for deep learning.
    - torchvision: PyTorch package for vision-related tasks.
    - models: Module containing pre-defined model architectures from torchvision.

Parameters and Settings:
    - root_path: Root directory for data storage and retrieval.
    - benchmark_out_path: Path to save the trained model.
    - prediction_out_path: Path to save prediction masks.
    - overlaid_out_path: Path to save overlaid images.
    - device: Device for computation (GPU or CPU).
    - input_size: Tuple specifying the input image size.
    - batch_size: Batch size for training and evaluation.
    - learning_rate: Learning rate for optimization.
    - epochs: Number of training epochs.
    - inital_epoch_loss: Initial loss value for epoch comparison.
    - num_update_lr: Number of epochs before updating learning rate.
    - test_size: Proportion of data for testing.
    - mask_train_path: Path to training mask data.
    - image_train_path: Path to training image data.
    - mask_test_path: Path to testing mask data.
    - image_test_path: Path to testing image data.
    - image_class: Specific image class for analysis.
    - explained_model: Pre-trained model for explanations.
    - transformations: Image transformations for preprocessing.
    - gradcam_layer: Layer for GradCAM explanations.
    - XAI_methods: Set of explanation methods to use.
    - aggregation_method: Method for aggregating explanations.

Note: Adjust the parameters and settings as needed to match your dataset and analysis requirements.
"""

import torch
from torchvision import models

root_path = "."

benchmark_out_path = "cenet_XAI_model.th"
benchmark_prediction_out_path = "out_XAI/masks-benchmark/"
cross_attention_out_path = "cross_attention_model.th"
cross_attention_prediction_out_path = "out_XAI/masks-cross-attention"
overlaid_out_path = "out_XAI/overlaid/"
vision_transformer_out_path = "vision_transformer_model.th"
vision_transformer_prediction_out_path = "out_XAI/masks-transformer-vision"
gated_fusion_out_path = "gated_fusion_model.th"
gated_fusion_prediction_out_path = "out_XAI/masks-gated-fusion"
hybrid_transformer_out_path = "hybrid_transformer_model.th"
hybrid_transformer_prediction_out_path = "out_XAI/masks-hybrid-transformer"
late_fusion_out_path = "late_fusion_model.th"
late_fusion_prediction_out_path = "out_XAI/masks-late_fusion"
mixture_of_experts_out_path = "mixture_of_experts.th"
mixture_of_experts_prediction_out_path = "out_XAI/masks-mixture-of-experts"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
input_size = (224, 224)
batch_size = 1
learning_rate = 0.0002
epochs = 10
inital_epoch_loss = 10000
num_update_lr = 20
test_size = 0.20

# Define database and class paths
mask_train_path = "/content/masks/train"
image_train_path = "/content/images/train"

mask_validation_path = "/content/masks/validation"
image_validation_path = "/content/images/validation"

mask_test_path = "/content/masks/test"
image_test_path = "/content/images/test"

image_class = "melanoma"  # Class for analysis (e.g., 'monkeys' n02483362 or 'water tower' n04562935)

image_classes = ["melanoma", "nevus", "seborrheic_keratosis"]

# Define explained model and transformations
explained_model = models.squeezenet1_1(
    weights="IMAGENET1K_V1"
)  # Choose from available models

transformations = (
    models.SqueezeNet1_1_Weights.DEFAULT.transforms()
)  # Choose from available transformations

# Define GradCAM layer and explanation methods
gradcam_layer = "classifier.1"  # Choose appropriate layer
XAI_methods = ["IntegratedGradients", "DeepLift"] # Set of explanation methods to use or 'nothing' for normal image segmentation

# Define aggregation method for explanations
aggregation_method = "concat"  # Choose from 'sum', 'channel', 'concat'

# Usage: Modify the values above to configure the image analysis pipeline and explanation methods.
