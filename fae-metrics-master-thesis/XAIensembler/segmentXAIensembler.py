#!/usr/bin/env python
# coding: utf-8

# **Importing Required Libraries**

import params
import augm
import modelarch
import thesisArchitecture_cnn
import thesisArchitecture_ViT
import thesisArchitecture_GatedFusion
import thesisArchitecture_HybridTransformer
import thesisArchitecture_LateFusion
import thesisArchitecture_MoE


from sklearn.metrics import auc

import metrics

import numpy as np
import cv2
import os
import time
import sys
from os.path import exists
from functools import reduce

from tqdm.notebook import tqdm

import torch
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
import captum.attr as capt
from PIL import Image
from skimage import segmentation

from matplotlib.colors import LinearSegmentedColormap

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("debug.log"), logging.StreamHandler(sys.stdout)],
)

all_available_methods = [
    "IntegratedGradients",
    "Saliency",
    "DeepLift",
    "DeepLiftShap",
    "GradientShap",
    "InputXGradient",
    "GuidedBackprop",
    "GuidedGradCam",
    "Deconvolution",
    "Occlusion",
    "ShapleyValueSampling",
    "Lime",
    "KernelShap",
    "LRP",
    "NoiseTunnel",
    "GradCam",
]


def get_model_module_by_name(module, access_string):
    """
    Retrieves a nested module or attribute within a module using dot-separated access.

    Args:
        module (module): The parent module to start accessing from.
        access_string (str): Dot-separated string specifying the path to the desired module or attribute.

    Returns:
        accessed_module: The accessed nested module or attribute.
    """
    names = access_string.split(sep=".")
    return reduce(getattr, names, module)


# **Data Loading and Preprocessing**

classes_dict = {
    "0": "melanoma",
    "1": "seborrheic_keratosis",
    "2": "nevus"
}

class ISIC_Dataset(Dataset):
    """
    Custom PyTorch Dataset class for Isic dataset images and explanations generation.

    Attributes:
        - params: Parameters for the dataset and explanations generation.
        - root: Root path of the dataset.
        - phase: Dataset phase (train, valid, or test).
        - images: List of image file paths.
        - labels: List of corresponding label/mask file paths.
        - model: Trained model for explanations generation.
        - model_transform: Image transformations for the model.
        - explain_methods: List of configured attribution methods.

    Methods:
        - generate_explanations(image, preds=None): Generates explanations for an image.
        - loader(img_path, mask_path, phase): Loads and preprocesses an image and mask.
        - read_dataset(root_path, mode): Reads and returns image and label/mask file paths.
        - __getitem__(index): Returns data for a given index.
        - __len__(): Returns the number of images in the dataset.
    """

    def __init__(self, params, root_path, phase="train"):
        """
        Initialize the IsicDataset.

        Args:
            - params: Parameters for the dataset and explanations generation.
            - root_path: Root path of the dataset.
            - phase: Dataset phase (train, valid, or test).
        """

        self.params = params
        self.root = root_path
        self.phase = phase
        self.images, self.labels = self.read_dataset(self.root, self.phase)
        self.model = params.explained_model.to(params.device)
        self.model.eval()
        # self.attribution_layer = params.explained_model
        self.model_transform = params.transformations

    def generate_explanations(self, image, preds=None):
        """
        Generate explanations for an input image.

        Args:
            - image: Input image for explanations generation.
            - preds: Predicted class label for the image (optional).

        Returns:
            List of generated explanations using various attribution methods.
        """

        segment = torch.from_numpy(
            segmentation.slic(image, n_segments=70, start_label=0)
        ).to(params.device)
        image = Image.fromarray(image)
        proper_data = self.model_transform(image).unsqueeze(dim=0).to(params.device)
        if preds is None:
            print("No prediction has been passed, using params.image_class")
            preds = [
                i for i, v in classes_dict.items() if v == params.image_class
            ][0]
        preds = torch.tensor([int(preds)]).to(params.device)

        xai_list = []
        rand_img_dist = torch.cat([proper_data * 0, proper_data * 1])

        if "IntegratedGradients" in params.XAI_methods:
            start_time = time.time()
            integrated_gradients = capt.IntegratedGradients(self.model)
            xai_list.append(
                integrated_gradients.attribute(proper_data, target=preds, n_steps=200)
                .squeeze()
                .cpu()
                .numpy()
            )
            logging.debug(
                "IntegratedGradients --- %s seconds ---" % (time.time() - start_time)
            )

        if "DeepLiftShap" in params.XAI_methods:
            start_time = time.time()
            deepliftshap = capt.DeepLiftShap(self.model)
            xai_list.append(
                deepliftshap.attribute(
                    proper_data, baselines=rand_img_dist, target=preds
                )
                .detach()
                .squeeze()
                .cpu()
                .numpy()
            )
            del deepliftshap
            logging.debug(
                "DeepLiftShap --- %s seconds ---" % (time.time() - start_time)
            )

        if "GradientShap" in params.XAI_methods:
            start_time = time.time()
            gradientshap = capt.GradientShap(self.model)
            xai_list.append(
                gradientshap.attribute(
                    proper_data, baselines=rand_img_dist, target=preds
                )
                .squeeze()
                .cpu()
                .numpy()
            )
            del gradientshap
            logging.debug(
                "GradientShap --- %s seconds ---" % (time.time() - start_time)
            )

        if "Occlusion" in params.XAI_methods:
            start_time = time.time()
            occlusion = capt.Occlusion(self.model)
            xai_list.append(
                occlusion.attribute(
                    proper_data,
                    target=preds,
                    strides=(3, 8, 8),
                    sliding_window_shapes=(3, 15, 15),
                    perturbations_per_eval=8,
                    baselines=0,
                )
                .squeeze()
                .cpu()
                .numpy()
            )
            del occlusion
            logging.debug("Occlusion --- %s seconds ---" % (time.time() - start_time))

        if "Lime" in params.XAI_methods:
            start_time = time.time()
            lime = capt.Lime(self.model)
            xai_list.append(
                lime.attribute(
                    proper_data,
                    target=preds,
                    feature_mask=segment,
                    perturbations_per_eval=8,
                )
                .squeeze()
                .cpu()
                .numpy()
            )
            del lime
            logging.debug("Lime --- %s seconds ---" % (time.time() - start_time))

        if "KernelShap" in params.XAI_methods:
            start_time = time.time()
            kernelshap = capt.KernelShap(self.model)
            xai_list.append(
                kernelshap.attribute(
                    proper_data, target=preds, feature_mask=segment, n_samples=25
                )
                .squeeze()
                .cpu()
                .numpy()
            )
            del kernelshap
            logging.debug("KernelShap --- %s seconds ---" % (time.time() - start_time))

        if "ShapleyValueSampling" in params.XAI_methods:
            start_time = time.time()
            shapleyvaluesampling = capt.ShapleyValueSampling(self.model)
            xai_list.append(
                shapleyvaluesampling.attribute(
                    proper_data, target=preds, feature_mask=segment
                )
                .squeeze()
                .cpu()
                .numpy()
            )
            del shapleyvaluesampling
            logging.debug(
                "ShapleyValueSampling --- %s seconds ---" % (time.time() - start_time)
            )

        if "GuidedGradCam" in params.XAI_methods:
            start_time = time.time()
            # change if different model!
            guidedgradcam = capt.GuidedGradCam(
                self.model, get_model_module_by_name(self.model, params.gradcam_layer)
            )
            xai_list.append(
                guidedgradcam.attribute(proper_data, target=preds)
                .detach()
                .cpu()
                .squeeze()
                .numpy()
            )
            del guidedgradcam
            logging.debug(
                "GuidedGradCam --- %s seconds ---" % (time.time() - start_time)
            )

        if "GradCam" in params.XAI_methods:
            start_time = time.time()
            gradcam = capt.LayerGradCam(
                self.model, get_model_module_by_name(self.model, params.gradcam_layer)
            )  # change if different model!
            gradcam = gradcam.attribute(proper_data, target=preds)
            gradcam = capt.LayerAttribution.interpolate(gradcam, params.input_size)
            gradcam = np.repeat(
                gradcam.detach().cpu().squeeze().numpy()[np.newaxis, ...], 3, axis=0
            )
            xai_list.append(gradcam)
            logging.debug("GradCam --- %s seconds ---" % (time.time() - start_time))

        if "NoiseTunnel" in params.XAI_methods:
            start_time = time.time()
            try:
                integrated_gradients
            except NameError:
                integrated_gradients = capt.IntegratedGradients(self.model)
            noise_tunnel = capt.NoiseTunnel(integrated_gradients)
            xai_list.append(
                noise_tunnel.attribute(proper_data, target=preds)
                .squeeze()
                .cpu()
                .numpy()
            )
            logging.debug("NoiseTunnel --- %s seconds ---" % (time.time() - start_time))

        if "Saliency" in params.XAI_methods:
            start_time = time.time()
            saliency = capt.Saliency(self.model)
            xai_list.append(
                saliency.attribute(proper_data, target=preds).squeeze().cpu().numpy()
            )
            del saliency
            logging.debug("Saliency --- %s seconds ---" % (time.time() - start_time))

        if "DeepLift" in params.XAI_methods:
            start_time = time.time()
            deeplift = capt.DeepLift(self.model)
            xai_list.append(
                deeplift.attribute(proper_data, target=preds)
                .detach()
                .squeeze()
                .cpu()
                .numpy()
            )
            del deeplift
            logging.debug("DeepLift --- %s seconds ---" % (time.time() - start_time))

        if "InputXGradient" in params.XAI_methods:
            start_time = time.time()
            inputxgradient = capt.InputXGradient(self.model)
            xai_list.append(
                inputxgradient.attribute(proper_data, target=preds)
                .detach()
                .cpu()
                .squeeze()
                .numpy()
            )
            del inputxgradient
            logging.debug(
                "InputXGradient --- %s seconds ---" % (time.time() - start_time)
            )

        if "GuidedBackprop" in params.XAI_methods:
            start_time = time.time()
            guidedbackprop = capt.GuidedBackprop(self.model)
            xai_list.append(
                guidedbackprop.attribute(proper_data, target=preds)
                .squeeze()
                .cpu()
                .numpy()
            )
            del guidedbackprop
            logging.debug(
                "GuidedBackprop --- %s seconds ---" % (time.time() - start_time)
            )

        if "Deconvolution" in params.XAI_methods:
            start_time = time.time()
            deconvolution = capt.Deconvolution(self.model)
            xai_list.append(
                deconvolution.attribute(proper_data, target=preds)
                .squeeze()
                .cpu()
                .numpy()
            )
            del deconvolution
            logging.debug(
                "Deconvolution --- %s seconds ---" % (time.time() - start_time)
            )

        if "LRP" in params.XAI_methods:
            start_time = time.time()
            lrp = capt.LRP(self.model)
            xai_list.append(
                lrp.attribute(proper_data, target=preds)
                .detach()
                .cpu()
                .squeeze()
                .numpy()
            )
            del lrp
            logging.debug("LRP --- %s seconds ---" % (time.time() - start_time))
        if len(xai_list) == 0:  # normal segmentation
            return_image = proper_data.squeeze().numpy()
        else:
            # elif params.aggregation_method == 'concat' or params.aggregation_method == 'sum' or params.aggregation_method == 'channel':
            return_image = np.concatenate(xai_list, axis=0)

        return return_image, preds

    def generate_explanation_for_batch_of_images(self, images, XAI_method, preds=None):
        """
        Generate explanations for a batch of images.

        Args:
            images (numpy array): Batch of images with shape (batch_size, 3, H, W) where H=224, W=224.
            XAI_method (str): The explainability method to use ("DeepLift" or "IntegratedGradients").
            preds (list or torch.Tensor, optional): Target predictions for each image. If None, default class will be used.

        Returns:
            numpy array: Explanations for the batch of images.
        """

        # Convert numpy images to torch tensors
        processed_images = []
        for img in images:
            # Check if image is in channel-first format (3, H, W)
            if img.shape[0] == 3:
                img = img.transpose(1, 2, 0)  # convert to (H, W, 3)
            img_pil = Image.fromarray(img.astype('uint8'))
            img_tensor = params.transformations(img_pil).unsqueeze(dim=0).to(params.device)
            processed_images.append(img_tensor)

        # Stack tensors into a batch
        proper_data = torch.cat(processed_images, dim=0)
        preds = torch.tensor(preds).to(params.device)

        if XAI_method == "DeepLift":
            torch.cuda.empty_cache()
            return capt.DeepLift(self.model).attribute(proper_data, target=preds).detach().squeeze().cpu().numpy()

        if XAI_method == "IntegratedGradients":
            torch.cuda.empty_cache()
            return capt.IntegratedGradients(self.model).attribute(proper_data, target=preds,
                                                             n_steps=200).detach().squeeze().cpu().numpy()


    def loader(self, img_path, mask_path, phase):
        """
        Load and preprocess image and mask data for a given phase (train, validation, test).

        Args:
            img_path (str): Path to the input image file.
            mask_path (str): Path to the mask image file.
            phase (str): The current phase of processing ('train', 'validation', or 'test').

        Returns:
            img (numpy.ndarray): Preprocessed input image data.
            mask (numpy.ndarray): Preprocessed mask image data.
            preds (list): List of predictions generated during explanation generation.
            expl (numpy.ndarray): Preprocessed explanation data.

        Notes:
            This function loads the input image and mask, and applies various preprocessing
            techniques depending on the phase. For the 'train' phase, data augmentation is
            applied using hue, saturation, value shifts, random shifts, scaling, rotation,
            flipping, and random 90-degree rotation. For other phases, no augmentation is
            performed.

            The input image is then used to generate explanations using a separate method,
            and the resulting explanations are preprocessed by normalizing them and
            converting them to floating-point values.

            The mask is loaded and normalized to the range [0, 1]. Pixel values greater
            than or equal to 0.5 are thresholded to 1, and values below 0.5 are thresholded
            to 0.

            The preprocessed image, mask, predictions, and explanations are returned as
            output.
        """

        img = cv2.imread(img_path)
        img = cv2.resize(img, params.input_size)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, params.input_size)

        if phase == "train":
            img = augm.randomHueSaturationValue(
                img,
                hue_shift_limit=(-30, 30),
                sat_shift_limit=(-5, 5),
                val_shift_limit=(-15, 15),
            )

            img, mask = augm.randomShiftScaleRotate(
                img,
                mask,
                shift_limit=(-0.1, 0.1),
                scale_limit=(-0.1, 0.1),
                aspect_limit=(-0.1, 0.1),
                rotate_limit=(-0, 0),
            )
            img, mask = augm.randomFlip(img, mask)
            img, mask = augm.randomRotate90(img, mask)

        predicted_class = [
            i for i, v in classes_dict.items() if img_path.__contains__(v)
        ][0]

        expl, preds = self.generate_explanations(img, predicted_class)

        expl = np.array(expl, np.float32) / np.max(expl)  # /255.0 #
        mask = np.array(mask, np.float32)
        mask = mask[np.newaxis, :, :]
        mask = (mask - np.min(mask)) / (np.max(mask) - np.min(mask))
        mask[mask >= 0.5] = 1
        mask[mask < 0.5] = 0

        return img, mask, preds, expl

    def read_dataset(self, root_path, mode="train"):
        """
        Reads and prepares a dataset for training or evaluation.

        Args:
            root_path (str): Root directory of the dataset.
            mode (str, optional): Mode for dataset loading, either 'train', 'valid', or 'test'.
                                Defaults to 'train'.

        Returns:
            images (list): List of file paths to the images.
            masks (list): List of file paths to the corresponding masks.
        """

        images = []
        masks = []
        for image_class in params.image_classes:
            if mode == "train":
                image_root = os.path.join(params.image_train_path, image_class)
                gt_root = os.path.join(params.mask_train_path, image_class)
            elif mode == "valid":
                image_root = os.path.join(params.image_validation_path, image_class)
                gt_root = os.path.join(params.mask_validation_path, image_class)
            else:
                image_root = os.path.join(params.image_test_path, image_class)
                gt_root = os.path.join(params.mask_test_path, image_class)

            tmp_list_of_images = os.listdir(image_root)

            list_of_images = []
            for image_name in tmp_list_of_images:
                image_path = os.path.join(image_root, image_name.split(".")[0] + ".jpg")
                label_path = os.path.join(gt_root, image_name.split(".")[0] + "_segmentation.png")

                if exists(image_path) and exists(label_path):
                    list_of_images.append(image_name)

            for image_name in list_of_images:
                image_path = os.path.join(image_root, image_name.split(".")[0] + ".jpg")
                label_path = os.path.join(gt_root, image_name.split(".")[0] + "_segmentation.png")

                if exists(image_path) and exists(label_path):
                    images.append(image_path)
                    masks.append(label_path)

        return images, masks

    def __getitem__(self, index):
        """
        Retrieve the data for a specific index in the dataset.

        Parameters:
        index (int): Index of the sample to retrieve.

        Returns:
        tuple: A tuple containing the explanation, mask, predictions, and image data (or relevant components).

        Note:
        The components of the returned tuple vary based on the dataset phase.
        """

        img, mask, preds, expl = self.loader(
            self.images[index], self.labels[index], self.phase
        )
        expl = torch.tensor(expl.copy(), dtype=torch.float32)
        mask = torch.tensor(mask.copy(), dtype=torch.float32)
        return expl, mask, preds, img.copy()

    def __len__(self):
        """
        Return the total number of samples in the dataset.

        Returns:
        int: Total number of samples in the dataset.
        """

        assert len(self.images) == len(
            self.labels
        ), "The number of images must be equal to labels"
        return len(self.images)


def plot_explanations(images, explanations, columns_names, classes_predicted):
    """
    Plot image explanations using heat maps.

    This function takes a list of images, their corresponding explanations,
    column names, and predicted classes, and generates a grid of subplots
    displaying the images along with heat map visualizations of their explanations.

    Parameters:
    images (list of ndarrays): List of input images (2D or 3D arrays).
    explanations (ndarray): Array of explanation maps for the images.
                            Each explanation map should have the shape (channels, height, width).
                            It is assumed that explanations for each image are divided into
                            consecutive groups of three channels in the explanations array.
    columns_names (list of str): List of column names for the subplot grid.
    classes_predicted (list of str): List of predicted classes for each image.

    Returns:
    None

    Dependencies:
    - numpy
    - matplotlib.pyplot
    - capt.visualization (Assuming this module provides the visualize_image_attr function)

    Note:
    The number of explanation maps should be a multiple of three, as each image's
    explanations are assumed to be grouped in threes (e.g., [explanation1, explanation2, explanation3]
    for the first image, [explanation4, explanation5, explanation6] for the second image, and so on).

    The code assumes a LinearSegmentedColormap named 'rg' with colors ["r", "w", "g"].

    Example usage:
    images = [image1, image2, ...]
    explanations = np.array([explanation1, explanation2, explanation3, explanation4, explanation5, ...])
    columns_names = ['Column 1', 'Column 2', ...]
    classes_predicted = ['Class A', 'Class B', ...]
    plot_explanations(images, explanations, columns_names, classes_predicted)
    """

    # Set up the default colormap
    default_cmap = LinearSegmentedColormap.from_list("rg", ["r", "w", "g"], N=256)

    # Generate a list of colormaps for each group of three explanations
    cmaps = [default_cmap] * (explanations.shape[0] // 3)

    # Set the sign for positive attributions
    sign = ["all"] * (explanations.shape[0] // 3)  # Positive attributions

    # Calculate the number of rows and columns for the subplot grid
    nrow, ncol = len(images) - 1, explanations.shape[0] // 3 + 2

    # Create a figure and axes for subplots
    fig, ax = plt.subplots(
        nrows=nrow,
        ncols=ncol,
        figsize=(1.5 * explanations.shape[0], 3 * explanations.shape[0]),
    )

    # Set subplot titles to column names
    for col, col_name in zip(ax, columns_names):
        col.title.set_text(col_name)

    # Iterate through images and explanations to create subplots
    for i, img in enumerate(images):
        ax[i].xaxis.set_ticks_position("none")
        ax[i].yaxis.set_ticks_position("none")
        ax[i].set_yticklabels([])
        ax[i].set_xticklabels([])

        # Display grayscale or RGB image based on its shape
        if img.shape[0] == 1:
            ax[i].imshow(np.array(img).transpose(1, 2, 0), vmin=0.0, vmax=1.0)
        else:
            ax[i].imshow(np.array(img), vmin=0, vmax=255)
            ax[i].set_ylabel(classes_predicted[i], size="large")

        # Generate heat map visualizations for explanations
        for j, col in enumerate(ax[2:]):
            expl = explanations[3 * j : 3 * (j + 1)]
            _ = capt.visualization.visualize_image_attr(
                expl.transpose(1, 2, 0),
                original_image=np.array(img),
                method="heat_map",
                sign=sign[j],
                plt_fig_axis=(fig, col),
                show_colorbar=True,
                outlier_perc=2,
                cmap=cmaps[j],
                use_pyplot=False,
            )

    torch.cuda.empty_cache()
    # Display the plot and save it as an image
    plt.savefig("plot_explanations.png", bbox_inches="tight", pad_inches=0)
    plt.show()


def plot_aggregated_explanations(
    params, images, explanations, columns_names, class_number, model = "Benchmark"
):
    """
    Plots aggregated explanations for image classification results.

    Parameters:
    - params (object): An object containing various parameters for the visualization.
    - images (list): A list of input images for which explanations are generated.
    - explanations (numpy.ndarray): A 4D numpy array containing the explanations for each image.
    - columns_names (list): A list of column names for different explanation components.
    - class_number (torch.Tensor): A tensor containing predicted class indices.
    - model (String): Model used ("Benchmark", "CrossAttention" or "VisionTransformer"

    Returns:
    - x (numpy.ndarray): The input image data.
    - y (numpy.ndarray): The predicted class indices.
    - s (numpy.ndarray): The mask used for explanations.
    - a (dict): A dictionary containing different explanation components.

    Note:
    - This function assumes the availability of various functions and libraries such as matplotlib, numpy, cv2,
      captum, and a model architecture defined in `modelarch`.

    Example usage:
    x, y, s, a = plot_aggregated_explanations(params, images, explanations, columns_names, class_number)
    """

    # if model == "Benchmark":
    model_name = params.benchmark_out_path
    prediction_out_path = params.benchmark_prediction_out_path

    if model == "CrossAttention":
        model_name = params.cross_attention_out_path
        prediction_out_path = params.cross_attention_prediction_out_path

    if model == "VisionTransformer":
        model_name = params.vision_transformer_out_path
        prediction_out_path = params.vision_transformer_prediction_out_path

    if model == "GatedFusion":
        model_name = params.gated_fusion_out_path
        prediction_out_path = params.gated_fusion_prediction_out_path

    if model == "HybridTransformer":
        model_name = params.hybrid_transformer_out_path
        prediction_out_path = params.hybrid_transformer_prediction_out_path

    if model == "LateFusion":
        model_name = params.late_fusion_out_path
        prediction_out_path = params.late_fusion_prediction_out_path

    if model == "MixtureOfExperts":
        model_name = params.mixture_of_experts_out_path
        prediction_out_path = params.mixture_of_experts_prediction_out_path

    classes_predicted = [classes_dict[str(i.item())] for i in class_number]
    y = class_number.cpu().numpy()

    a = {}

    cmaps = [None] * (explanations.shape[0] // 3)
    sign = ["all"] * (explanations.shape[0] // 3)  # positive
    nrow, ncol = len(images) - 1, explanations.shape[0] // 3 + 4
    fig, ax = plt.subplots(
        nrows=nrow,
        ncols=ncol,
        figsize=(1.5 * explanations.shape[0] * 2, 3 * explanations.shape[0] * 2),
    )
    for col, col_name in zip(ax, columns_names):
        col.title.set_text(col_name)

    # Loop over images and generate explanations
    for i, img in enumerate(images[:-1]):
        start_time = time.time()

        ax[i].xaxis.set_ticks_position("none")
        ax[i].yaxis.set_ticks_position("none")
        ax[i].set_yticklabels([])
        ax[i].set_xticklabels([])

        # Plot original image
        ax[i].imshow(np.array(img)[..., ::-1], vmin=0, vmax=255)
        ax[i].set_ylabel(classes_predicted[i], size="large")
        x = np.array(img)[..., ::-1]

        # Plot mask
        ax[i + 1].xaxis.set_ticks_position("none")
        ax[i + 1].yaxis.set_ticks_position("none")
        ax[i + 1].set_yticklabels([])
        ax[i + 1].set_xticklabels([])
        mask = np.array(images[-1]).transpose(1, 2, 0)
        ax[i + 1].imshow(mask, vmin=0.0, vmax=1.0)
        s = mask

        if not os.path.exists(prediction_out_path):
            os.makedirs(prediction_out_path)
        cv2.imwrite(
            os.path.join(
                prediction_out_path,
                "mask-{}-{}.png".format(classes_predicted[i], start_time),
            ),
            np.array(images[-1]).transpose(1, 2, 0) * 255,
        )

        # Generate predicted mask

        # if model == "Benchmark":
        ensemble_model = modelarch.CE_Net_(params).to(params.device)
        if model == "CrossAttention":
            ensemble_model = thesisArchitecture_cnn.EnsembleExplanationNetwork().to(params.device)
        if model == "VisionTransformer":
            ensemble_model = thesisArchitecture_ViT.EnsembleExplanationNetwork(params).to(params.device)
        if model == "GatedFusion":
            ensemble_model = thesisArchitecture_GatedFusion.EnsembleExplanationNetwork(params).to(params.device)
        if model == "HybridTransformer":
            ensemble_model = thesisArchitecture_HybridTransformer.EnsembleExplanationNetwork(params).to(params.device)
        if model == "LateFusion":
            ensemble_model = thesisArchitecture_LateFusion.EnsembleExplanationNetwork(params).to(params.device)
        if model == "MixtureOfExperts":
            ensemble_model = thesisArchitecture_MoE.EnsembleExplanationNetwork(params).to(params.device)

        weights = torch.load(model_name, map_location=torch.device(params.device))
        ensemble_model.load_state_dict(weights)

        tmp_explanations = torch.tensor(
            np.expand_dims(explanations, axis=0), dtype=torch.float32
        ).to(params.device)
        pred, ensemble_expl = ensemble_model.forward(params, tmp_explanations)
        print("Prediction max value:", pred.max())
        print("Prediction min value:", pred.min())
        print("Explanation max value:", ensemble_expl.max())
        print("Explanation min value:", ensemble_expl.min())

        pred = torch.round(pred)
        pred = np.squeeze(pred.detach().cpu().numpy(), axis=0)
        pred = pred.transpose(1, 2, 0)
        ax[i + 2].xaxis.set_ticks_position("none")
        ax[i + 2].yaxis.set_ticks_position("none")
        ax[i + 2].set_yticklabels([])
        ax[i + 2].set_xticklabels([])
        ax[i + 2].imshow(pred, vmin=0.0, vmax=1.0)
        cv2.imwrite(
            os.path.join(
                prediction_out_path,
                "mask-predicted-{}-{}.png".format(classes_predicted[i], start_time),
            ),
            pred * 255,
        )

        # Plot all explanations
        for j, col in enumerate(ax[3:-1]):
            expl = explanations[3 * j : 3 * (j + 1)]

            _ = capt.visualization.visualize_image_attr(
                expl.transpose(1, 2, 0),
                original_image=np.array(img),
                method="heat_map",
                sign=sign[j],
                plt_fig_axis=(fig, col),
                show_colorbar=True,
                outlier_perc=2,
                cmap=cmaps[j],
                use_pyplot=False,
            )
            a[columns_names[3 + j]] = expl  # .transpose(1, 2, 0)

        # Plot ensembled explanation
        ensemble_expl = -1 + 2 * (ensemble_expl - torch.min(ensemble_expl)) / (
            torch.max(ensemble_expl) - torch.min(ensemble_expl)
        )
        ensemble_expl = np.squeeze(ensemble_expl.detach().cpu().numpy(), axis=0)
        ensemble_expl = ensemble_expl.transpose(1, 2, 0)
        capt.visualization.visualize_image_attr(
            ensemble_expl,
            original_image=np.array(img),
            method="heat_map",
            sign=sign[0],
            plt_fig_axis=(fig, ax[-1]),
            show_colorbar=True,
            outlier_perc=2,
            cmap=cmaps[0],
            use_pyplot=False,
        )

        cv2.imwrite(
            os.path.join(
                prediction_out_path,
                "ensembled-explanation-{}-{}.png".format(
                    classes_predicted[i], start_time
                ),
            ),
            ensemble_expl * 255,
        )
        a["EnsembledXAI"] = np.swapaxes(ensemble_expl, -1, 0)

    # Show the final plot

    plt.savefig(
        "{}overlaid-{}.png".format(params.overlaid_out_path, time.time()),
        bbox_inches="tight",
    )
    plt.show()

    torch.cuda.empty_cache()
    # Return necessary values
    return x, y, s, a


def retrieve_data(params, model = "Benchmark"):
    """
    Retrieve data from the Isic dataset, generate aggregated explanations, and prepare for visualization.

    Args:
        - params: Parameters for data retrieval and explanation generation.
        - model (String): Model used ("Benchmark", "CrossAttention" or "VisionTransformer"

    Returns:
        - x_batch: Batch of images prepared for visualization.
        - y_batch: Batch of corresponding mask images.
        - s_batch: Batch of predicted class labels for the images.
        - a_batch: Dictionary of aggregated explanations for different attribution methods.
    """

    root_path = params.root_path
    valid_dataset = ISIC_Dataset(params, root_path, phase="test")

    # Initialize column names for the aggregated explanations
    col_names = (
        ["Original", "Mask", "Predicted mask"]
        + list(filter(lambda x: x in params.XAI_methods, all_available_methods))
        + ["Ensembled XAI"]
    )

    example = valid_dataset[0]

    #if (model == "Benchmark")
    ensemble_model = modelarch.CE_Net_(params).to(params.device)
    if model == "CrossAttention":
        ensemble_model = thesisArchitecture_cnn.EnsembleExplanationNetwork().to(params.device)
    if model == "VisionTransformer":
        ensemble_model = thesisArchitecture_ViT.EnsembleExplanationNetwork(params).to(params.device)
    if model == "GatedFusion":
        ensemble_model = thesisArchitecture_GatedFusion.EnsembleExplanationNetwork(params).to(params.device)
    if model == "HybridTransformer":
        ensemble_model = thesisArchitecture_HybridTransformer.EnsembleExplanationNetwork(params).to(params.device)
    if model == "LateFusion":
        ensemble_model = thesisArchitecture_LateFusion.EnsembleExplanationNetwork(params).to(params.device)
    if model == "MixtureOfExperts":
        ensemble_model = thesisArchitecture_MoE.EnsembleExplanationNetwork(params).to(params.device)

    explanations = example[0].numpy()
    tmp_explanations = torch.tensor(
        np.expand_dims(explanations, axis=0), dtype=torch.float32
    ).to(params.device)
    pred, ensemble_expl = ensemble_model.forward(params, tmp_explanations)
    torch.cuda.empty_cache()

    x_batch = []
    y_batch = []
    s_batch = []

    # Initialize an empty dictionary to store aggregated explanations
    a_batch = {}

    # Loop through each example in the dataset
    for example in valid_dataset:
        print(len(example))
        # Generate aggregated explanations and retrieve relevant data
        x, y, s, a = plot_aggregated_explanations(
            params, [example[3], example[1]], example[0].numpy(), col_names, example[2], model
        )

        # Append data to respective batches
        x_batch.append(x)
        y_batch.append(y)
        s_batch.append(s)

        # Iterate through aggregated explanations and update the dictionary
        for k, v in a.items():
            v = np.expand_dims(v, axis=0)
            if k in a_batch:
                a_batch[k] = np.concatenate((a_batch[k], v), axis=0)
            else:
                a_batch[k] = v

    # Perform necessary data transformations and normalization
    x_batch = np.swapaxes(x_batch, -1, 1)
    x_batch = (x_batch - np.min(x_batch)) / (np.max(x_batch) - np.min(x_batch))
    y_batch = np.squeeze(y_batch)
    s_batch = np.squeeze(s_batch)

    torch.cuda.empty_cache()
    # Return the prepared batches of data and aggregated explanations
    return x_batch, y_batch, s_batch, a_batch


def evaluate_on_image(params, model, images, targets, device):
    """
    Evaluate model predictions and generate ensemble explanations for a batch of images.

    Args:
        - params: Parameters for dataset and ensemble explanations.
        - model: The trained model for evaluation and ensemble explanation.
        - images: Batch of input images to be evaluated and explained.
        - targets: Target class labels for the input images.
        - device: Device (e.g., CPU or GPU) for computation.

    Returns:
        Ensemble explanations for the batch of input images after evaluating the model.
    """

    root_path = params.root_path
    dataset = ISIC_Dataset(params, root_path, phase="valid")
    valid_dataset = []
    for i in range(images.numpy().shape[0]):
        img = np.swapaxes(images[i, ...].numpy(), 0, -1) * 255
        expl, preds = dataset.generate_explanations(img.astype(np.uint8), targets[i])
        expl = np.array(expl, np.float32) / np.max(expl)
        valid_dataset.append(expl)

    example = valid_dataset[0]

    # Initialize an empty list to store ensemble explanations
    a_batch = []

    # Loop through each example in the batch
    for example in valid_dataset:
        # Prepare the individual explanation for the model's forward pass
        tmp_explanations = torch.tensor(
            np.expand_dims(example, axis=0), dtype=torch.float32
        ).to(device)

        # Forward pass through the model to get ensemble explanations
        _, ensemble_expl = model.forward(params, tmp_explanations)
        ensemble_expl = np.squeeze(ensemble_expl.detach().cpu().numpy(), axis=0)

        # Append the ensemble explanation to the batch
        a_batch.append(ensemble_expl)

    # Convert the batch of ensemble explanations to a numpy array
    a_batch = np.asarray(a_batch)

    # Clear GPU memory cache
    torch.cuda.empty_cache()

    # Return the batch of ensemble explanations
    return a_batch


def show_example(params, valid_dataset, col_names):
    """
    Visualizes explanations for a given example from the validation dataset.

    Parameters:
    - params (object): An object containing parameters for explanation methods and visualization.
    - valid_dataset (list): A dataset containing validation examples, where each example is a tuple.
                           The tuple structure should be (image, mask, predicted_labels, explanations).
    - col_names (list): A list of column names for the visualization table, including "Original", "Mask",
                       and names of explanation methods used in params.XAI_methods.

    Returns:
    None

    This function extracts an example from the validation dataset, retrieves the predicted class labels,
    and visualizes the explanations generated by different explanation methods. The visualization
    includes the original image, the mask applied to the image, and the explanations provided by
    various XAI (Explainable AI) methods specified in params.XAI_methods.

    Example usage:
    params = SomeParametersObject()
    valid_dataset = [(image1, mask1, predicted_labels1, explanations1), (image2, mask2, predicted_labels2, explanations2), ...]
    col_names = ["Original", "Mask", "LIME", "SHAP", "GradCAM"]
    show_example(params, valid_dataset, col_names)
    """
    # Extract the first example from the validation dataset
    example = valid_dataset[0]

    # Create column names for explanations
    col_names = ["Original", "Mask"] + list(
        filter(lambda x: x in params.XAI_methods, all_available_methods)
    )

    # Extract predicted class labels
    predicted_names = [classes_dict[str(i.item())] for i in example[2]]

    # Plot the explanations using a custom function (plot_explanations)
    plot_explanations(
        [example[3], example[1]], example[0].numpy(), col_names, predicted_names
    )


def train(params, train_loader, train_dataset, valid_loader, valid_dataset, col_names, model = "Benchmark"):
    """
    Train the neural network model using the given data loaders and configuration parameters.

    Args:
        params: Configuration parameters for training.
        train_loader: DataLoader for training dataset.
        train_dataset: Training dataset.
        valid_loader: DataLoader for validation dataset.
        valid_dataset: Validation dataset.
        col_names (list): List of column names for logging and visualization.
        model (String): "Benchmark", "CrossAttention" or "VisionTransformer"

    Returns:
        None
    """

    # if model = "Benchmark"
    solver = modelarch.MyFrame(params, params.learning_rate, params.device)
    solver_out_path = params.benchmark_out_path
    if model == "CrossAttention":
        solver = thesisArchitecture_cnn.MyFrame(params, params.learning_rate)
        solver_out_path = params.cross_attention_out_path
    if model == "VisionTransformer":
        solver = thesisArchitecture_ViT.MyFrame(params, params.learning_rate)
        solver_out_path = params.vision_transformer_out_path
    if model == "GatedFusion":
        solver = thesisArchitecture_GatedFusion.MyFrame(params, params.learning_rate)
        solver_out_path = params.gated_fusion_out_path
    if model == "HybridTransformer":
        solver = thesisArchitecture_HybridTransformer.MyFrame(params, params.learning_rate)
        solver_out_path = params.hybrid_transformer_out_path
    if model == "LateFusion":
        solver = thesisArchitecture_LateFusion.MyFrame(params, params.learning_rate)
        solver_out_path = params.late_fusion_out_path
    if model == "MixtureOfExperts":
        solver = thesisArchitecture_MoE.MyFrame(params, params.learning_rate, params.device)
        solver_out_path = params.mixture_of_experts_out_path

    no_optim = 0
    valid_epoch_best_loss = params.inital_epoch_loss
    Loss = []
    Accuracy = []
    Sensitivity = []
    Precision = []
    F1 = []
    IoU = []
    # how_frequently = 0
    epochs = params.epochs
    start_training_time = time.time()

    # Iterate over epochs
    for epoch in range(1, epochs + 1):
        logging.info("Epoch {}/{}".format(epoch, epochs))
        train_epoch_loss = 0
        train_epoch_acc = 0
        train_epoch_sen = 0
        train_epoch_prec = 0
        train_epoch_f1 = 0
        train_epoch_iou = 0

        valid_epoch_loss = 0
        valid_epoch_acc = 0
        valid_epoch_sen = 0
        valid_epoch_prec = 0
        valid_epoch_f1 = 0
        valid_epoch_iou = 0

        index = 0
        length = len(train_loader)
        iterator = tqdm(
            enumerate(train_loader),
            total=length,
            leave=False,
            desc=f"Epoch {epoch}/{epochs}",
        )
        for index, (img, mask, _, _) in iterator:
            img = img.to(params.device)
            mask = mask.to(params.device)
            solver.set_input(img, mask)
            train_loss, pred = solver.optimize()
            train_acc, train_sen, train_prec, train_f1, train_iou = metrics.acc_sen(
                pred, mask
            )

            train_epoch_loss += train_loss.detach().cpu().numpy()
            train_epoch_acc += train_acc.detach().cpu().numpy()
            train_epoch_sen += train_sen.detach().cpu().numpy()
            train_epoch_prec += train_prec.detach().cpu().numpy()
            train_epoch_f1 += train_f1.detach().cpu().numpy()
            train_epoch_iou += train_iou.detach().cpu().numpy()

        # Calculate average metrics for training epoch
        train_epoch_loss = train_epoch_loss / len(train_loader)
        train_epoch_acc = train_epoch_acc / len(train_dataset)
        train_epoch_sen = train_epoch_sen / len(train_dataset)
        train_epoch_prec = train_epoch_prec / len(train_dataset)
        train_epoch_f1 = train_epoch_f1 / len(train_dataset)
        train_epoch_iou = train_epoch_iou / len(train_dataset)

        # Log training metrics
        logging.debug("train_loss:", train_epoch_loss)
        logging.debug("train_accuracy:", train_epoch_acc)
        logging.debug("train_recall:", train_epoch_sen)
        logging.debug("train_precision:", train_epoch_prec)
        logging.debug("train_f1:", train_epoch_f1)
        logging.debug("train_iou:", train_epoch_iou)

        # Store metrics for visualization
        Loss.append(train_epoch_loss)
        Accuracy.append(train_epoch_acc)
        Sensitivity.append(train_epoch_sen)
        Precision.append(train_epoch_prec)
        F1.append(train_epoch_f1)
        IoU.append(train_epoch_iou)

        # Save the model
        solver.save(solver_out_path)

        # Validation
        length = len(valid_loader)
        iterator = tqdm(
            enumerate(valid_loader), total=length, leave=False, desc="Valid"
        )
        for index, (img, mask, _, _) in iterator:
            img = img.to(params.device)
            mask = mask.to(params.device)

            solver.set_input(img, mask)
            valid_loss, pred, ensemble_expl = solver.calculate_loss()
            valid_acc, valid_sen, valid_prec, valid_f1, valid_iou = metrics.acc_sen(
                pred, mask
            )

            valid_epoch_loss += valid_loss.detach().cpu().numpy()
            valid_epoch_acc += valid_acc.detach().cpu().numpy()
            valid_epoch_sen += valid_sen.detach().cpu().numpy()
            valid_epoch_prec += valid_prec.detach().cpu().numpy()
            valid_epoch_f1 += valid_f1.detach().cpu().numpy()
            valid_epoch_iou += valid_iou.detach().cpu().numpy()

        # Calculate average metrics for validation epoch
        valid_epoch_loss = valid_epoch_loss / len(valid_loader)
        valid_epoch_acc = valid_epoch_acc / len(valid_dataset)
        valid_epoch_sen = valid_epoch_sen / len(valid_dataset)
        valid_epoch_prec = valid_epoch_prec / len(valid_dataset)
        valid_epoch_f1 = valid_epoch_f1 / len(valid_dataset)
        valid_epoch_iou = valid_epoch_iou / len(valid_dataset)

        # Log validation metrics
        logging.debug("valid_loss:", valid_epoch_loss)
        logging.debug("valid_accuracy:", valid_epoch_acc)
        logging.debug("valid_recall:", valid_epoch_sen)
        logging.debug("valid_precision:", valid_epoch_prec)
        logging.debug("valid_f1:", valid_epoch_f1)
        logging.debug("valid_iou:", valid_epoch_iou)

        # Check if validation loss has improved
        if valid_epoch_loss >= valid_epoch_best_loss:
            no_optim += 1
        else:
            no_optim = 0
            valid_epoch_best_loss = valid_epoch_loss
            # Save the model if validation loss improves
            solver.save(model + "-{}-epoch{}.th".format(start_training_time, epoch))
            # Visualize explanations if conditions are met
            # if epoch > 19 and how_frequently > 9:
            #     how_frequently = 0
            #     for example in valid_dataset:
            #         plot_aggregated_explanations(params, [example[3], example[1]], example[0].numpy(), col_names, example[2])
        # how_frequently += 1

        # Update learning rate if no improvement in validation loss
        if no_optim > params.num_update_lr:
            if solver.lr < 1e-9:
                break
            solver.load(solver_out_path)
            solver.update_lr(2.0, factor=True)


def visualize_results(valid_dataset, col_names, model = "Benchmark"):
    """
    Visualize the final ensemble of XAI (Explainable AI) explanations.

    This function iterates through the examples in the validation dataset and generates aggregated explanations
    for each example. The aggregated explanations are then visualized using the 'plot_aggregated_explanations' function.

    Parameters:
    valid_dataset (Dataset): Validation dataset containing image examples and associated data.
    col_names (list of str): List of column names for visualization.
    model (String): "Benchmark", "CrossAttention", "VisionTransformer"

    Returns:
    None

    Dependencies:
    - plot_aggregated_explanations function

    Example usage:
    visualize_results(valid_dataset, col_names)
    """

    for example in valid_dataset:
        plot_aggregated_explanations(
            params, [example[3], example[1]], example[0].numpy(), col_names, example[2], model
        )


def explainer(model, inputs, targets, **kwargs) -> np.ndarray:
    """
    Generate an ensemble of XAI explanations for a given model and inputs.

    This function computes XAI explanations for a given model and input data using an ensemble approach.
    The model predictions are compared against the ground truth targets, and the XAI explanations are
    calculated based on the model's attributions to the input features.

    Parameters:
    model (torch.nn.Module): The neural network model to explain.
    inputs (numpy.ndarray): Input data (images) for which explanations are computed.
    targets (torch.Tensor): Ground truth targets for the input data.
    **kwargs: Additional keyword arguments.
             - device (str): Device on which to perform computations (default: 'cpu').

    Returns:
    a_batch (numpy.ndarray): An array of XAI explanations for each input in the batch.

    Dependencies:
    - evaluate_on_image function (Assuming this function evaluates the model on input data and returns attributions)

    Example usage:
    x_batch, y_batch, s_batch, a_batch = retrieve_data(params)
    model = ensemble model used for explanation generation
    attrs = EnsembledXAI_explainer(model=model, inputs=x_batch, targets=y_batch, **{"device": params.device"})
    """

    device = kwargs.get("device", "cpu")
    # img_size = kwargs.get("img_size", 224)
    # nr_channels = kwargs.get("nr_channels", 3)

    inputs = torch.from_numpy(inputs)
    a_batch = evaluate_on_image(params, model, inputs, targets, device)

    return a_batch

def evaluate_model(test_loader, model_name, params):
    """
    Evaluate the given model on a test dataset and return ground truth metrics.

    Args:
        test_loader (DataLoader): PyTorch DataLoader for the test dataset.
        model_name (str): Name of the model to evaluate.
        params: Parameter object containing model configurations and device settings.

    Returns:
        dict: A dictionary containing accuracy, sensitivity, precision, F1-score, and IoU.
    """

    # Load the appropriate model based on the provided model name
    if model_name == "Benchmark":
        model = modelarch.CE_Net_(params).to(params.device)
        model_path = params.benchmark_out_path
    elif model_name == "CrossAttention":
        model = thesisArchitecture_cnn.EnsembleExplanationNetwork().to(params.device)
        model_path = params.cross_attention_out_path
    elif model_name == "VisionTransformer":
        model = thesisArchitecture_ViT.EnsembleExplanationNetwork(params).to(params.device)
        model_path = params.vision_transformer_out_path
    elif model_name == "GatedFusion":
        model = thesisArchitecture_GatedFusion.EnsembleExplanationNetwork(params).to(params.device)
        model_path = params.gated_fusion_out_path
    elif model_name == "HybridTransformer":
        model = thesisArchitecture_HybridTransformer.EnsembleExplanationNetwork(params).to(params.device)
        model_path = params.hybrid_transformer_out_path
    elif model_name == "LateFusion":
        model = thesisArchitecture_LateFusion.EnsembleExplanationNetwork(params).to(params.device)
        model_path = params.late_fusion_out_path
    elif model_name == "MixtureOfExperts":
        model = thesisArchitecture_MoE.EnsembleExplanationNetwork(params).to(params.device)
        model_path = params.mixture_of_experts_out_path
    else:
        raise ValueError("Unknown model name: {}".format(model_name))

    # Load model weights
    model.load_state_dict(torch.load(model_path, map_location=params.device))
    model.eval()

    # Initialize metrics
    total_acc = 0
    total_sen = 0
    total_prec = 0
    total_f1 = 0
    total_iou = 0
    total_samples = 0

    # Evaluate the model
    with torch.no_grad():
        for img, mask, _, _ in test_loader:
            img, mask = img.to(params.device), mask.to(params.device)

            # Get model predictions
            output, _ = model.forward(params, img)
            pred = torch.round(output)

            # Compute metrics
            acc, sen, prec, f1, iou = metrics.acc_sen(pred, mask)

            total_acc += acc.item()
            total_sen += sen.item()
            total_prec += prec.item()
            total_f1 += f1.item()
            total_iou += iou.item()
            total_samples += 1

    # Compute averages
    avg_acc = total_acc / total_samples
    avg_sen = total_sen / total_samples
    avg_prec = total_prec / total_samples
    avg_f1 = total_f1 / total_samples
    avg_iou = total_iou / total_samples

    # Return metrics
    return {
        "Accuracy": avg_acc,
        "Sensitivity": avg_sen,
        "Precision": avg_prec,
        "F1-score": avg_f1,
        "IoU": avg_iou,
    }

def plot_roc_curve(test_loader, model_name, params):
    """
    Plot ROC curve for a segmentation model by varying threshold from 0 to 1 (50 points).

    Args:
        test_loader (DataLoader): Test data loader.
        model: Trained PyTorch model.
        params: Parameter object with device info.
    """
    thresholds = torch.linspace(0, 1, 50)

    # Load the appropriate model based on the provided model name
    if model_name == "Benchmark":
        model = modelarch.CE_Net_(params).to(params.device)
        model_path = params.benchmark_out_path
    elif model_name == "CrossAttention":
        model = thesisArchitecture_cnn.EnsembleExplanationNetwork().to(params.device)
        model_path = params.cross_attention_out_path
    elif model_name == "VisionTransformer":
        model = thesisArchitecture_ViT.EnsembleExplanationNetwork(params).to(params.device)
        model_path = params.vision_transformer_out_path
    elif model_name == "GatedFusion":
        model = thesisArchitecture_GatedFusion.EnsembleExplanationNetwork(params).to(params.device)
        model_path = params.gated_fusion_out_path
    elif model_name == "HybridTransformer":
        model = thesisArchitecture_HybridTransformer.EnsembleExplanationNetwork(params).to(params.device)
        model_path = params.hybrid_transformer_out_path
    elif model_name == "LateFusion":
        model = thesisArchitecture_LateFusion.EnsembleExplanationNetwork(params).to(params.device)
        model_path = params.late_fusion_out_path
    elif model_name == "MixtureOfExperts":
        model = thesisArchitecture_MoE.EnsembleExplanationNetwork(params).to(params.device)
        model_path = params.mixture_of_experts_out_path

    # Load model weights
    if model_name not in all_available_methods:
        model.load_state_dict(torch.load(model_path, map_location=params.device))
        model.eval()

    all_tpr = []
    all_fpr = []
    all_precision = []
    all_recall = []

    with torch.no_grad():
        for threshold in thresholds:
            tpr_list = []
            fpr_list = []
            precision_list = []
            recall_list = []
            for img, mask, _, _ in test_loader:
                img, mask = img.to(params.device), mask.to(params.device)

                if model_name not in all_available_methods:
                    # Base explanation
                    output, _ = model.forward(params, img)
                    pred = (output > threshold).float()
                else:
                    # Ensembled explanation
                    index = params.XAI_methods.index(model_name)
                    start = 3 * index
                    end = 3 * index + 3
                    pred = create_mask_from_explanation(img[:, start:end, :, :], threshold.item() * 100)

                # Flatten predictions and masks for easier metric computation
                pred_flat = pred.view(pred.size(0), -1)
                mask_flat = mask.view(mask.size(0), -1)

                # Calculate confusion matrix components
                TP = (pred_flat * mask_flat).sum(1)
                TN = ((1 - pred_flat) * (1 - mask_flat)).sum(1)
                FP = (pred_flat * (1 - mask_flat)).sum(1)
                FN = ((1 - pred_flat) * mask_flat).sum(1)

                # Compute metrics (add small constant to avoid division by zero)
                TPR = TP / (TP + FN + 1e-6)  # Sensitivity / Recall
                FPR = FP / (FP + TN + 1e-6)
                precision = TP / (TP + FP + 1e-6)
                recall = TPR  # Recall is equivalent to sensitivity

                tpr_list.append(TPR.mean().item())
                fpr_list.append(FPR.mean().item())
                precision_list.append(precision.mean().item())
                recall_list.append(recall.mean().item())

            # Average metrics over the test loader for current threshold
            all_tpr.append(np.mean(tpr_list))
            all_fpr.append(np.mean(fpr_list))
            all_precision.append(np.mean(precision_list))
            all_recall.append(np.mean(recall_list))

    # --- ROC Curve and AUC ---
    sorted_indices = np.argsort(all_fpr)
    print("FPR:", all_fpr)
    print("TPR:", all_tpr)
    all_fpr_sorted = np.array(all_fpr)[sorted_indices]
    all_tpr_sorted = np.array(all_tpr)[sorted_indices]

    roc_auc = auc(all_fpr_sorted, all_tpr_sorted)
    print("AUC ROC value:", roc_auc)

    plt.figure(figsize=(7, 6))
    plt.plot(all_fpr_sorted, all_tpr_sorted, marker='o', label="ROC Curve " + model_name)
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("ROC Curve for " + model_name + " model")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

    # --- Precision-Recall Curve and AUC ---
    sorted_indices_pr = np.argsort(all_recall)
    all_recall_sorted = np.array(all_recall)[sorted_indices_pr]
    all_precision_sorted = np.array(all_precision)[sorted_indices_pr]

    pr_auc = auc(all_recall_sorted, all_precision_sorted)
    print("AUC PR value:", pr_auc)

    plt.figure(figsize=(7, 6))
    plt.plot(all_recall_sorted, all_precision_sorted, marker='o', label="Precision-Recall Curve " + model_name)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve for " + model_name + " model")
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.show()


def create_mask_from_explanation(image, percentile=80):
    """
    Generate a binary mask highlighting important pixels based on an explanation map.

    Args:
        image (torch.Tensor): Tensor of shape (1, 3, 224, 224) containing explanation values.
        device (str): The device to return the mask on ('cpu' or 'cuda').

    Returns:
        torch.Tensor: Binary mask of shape (1, 1, 224, 224) with values {0, 1}.
    """
    image_np = image.detach().clone().cpu().squeeze(0).numpy()
    image_2d = np.mean(np.abs(image_np), axis=0)

    # Compute the percentile value
    threshold = np.percentile(image_2d, percentile)

    # Create binary mask where pixel importance > threshold
    mask_np = (image_2d > threshold).astype(np.uint8)  # shape: (224, 224)

    # Expand dimensions to (1, 1, 224, 224)
    mask_np = np.expand_dims(mask_np, axis=(0, 1))

    # Convert to torch tensor and move to desired device
    mask = torch.from_numpy(mask_np).to(params.device)

    return mask

def main(params):
    """
    Execute the main training and validation pipeline for a neural network model that ensembles XAI (Explainable AI) and provide its visualization.

    This function orchestrates the complete training and validation process, including dataset loading, training,
    validation, visualization of results, and an example validation using the Quantus library.

    Parameters:
    params (argparse.Namespace): A namespace object containing various parameters for configuration.

    Returns:
    None

    Dependencies:
    - ISIC_Dataset
    - DataLoader from torch.utils.data
    - logging module
    - show_example function
    - train function
    - visualize_results function
    - retrieve_data function
    - modelarch module
    - EnsembledXAI_explainer function

    Note:
    The functions and modules mentioned in the "Dependencies" section should be defined and imported properly.

    Example usage:
    Define a `params` object with required parameters, then call main(params) to start the pipeline.
    """
    # Define column names for the visualization
    col_names = (
        ["Original", "Mask", "Predicted mask"]
        + list(filter(lambda x: x in params.XAI_methods, all_available_methods))
        + ["Ensembled XAI"]
    )

    # Load the training dataset and create a data loader
    train_dataset = ISIC_Dataset(params, root_path=params.root_path, phase="train")
    train_loader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True)
    logging.info("Size of the train dataset : {}".format(len(train_dataset)))

    # Load the validation dataset and create a data loader
    valid_dataset = ISIC_Dataset(params, root_path=params.root_path, phase="valid")
    valid_loader = DataLoader(valid_dataset, batch_size=params.batch_size, shuffle=True)
    logging.info("Size of the valid dataset : {}".format(len(valid_dataset)))

    # Show an example validation with visualization
    show_example(params, valid_dataset, col_names)

    # Train the model using the training dataset and validate using the validation dataset
    train(params, train_loader, train_dataset, valid_loader, valid_dataset, col_names)

    # Visualize the results on the validation dataset
    visualize_results(valid_dataset, col_names)

    # Load image, label, mask, and explanation
    x_batch, y_batch, s_batch, a_batch = retrieve_data(params)

    # Load the ensemble model defined in params
    model = modelarch.CE_Net_(params).to(params.device)

    # Perform an example validation using the Quantus library
    attrs = explainer(
        model=model, inputs=x_batch, targets=y_batch, **{"device": params.device}
    )
