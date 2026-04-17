# Unsupervised Detection of Post-Stroke Brain Abnormalities

## Overview

This repository contains the evaluation code built upon **[REFLECT](https://github.com/farzad-bz/REFLECT)**, a state-of-the-art unsupervised brain anomaly segmentation framework based on rectified flows. The model is evaluated for its ability to detect post-stroke brain abnormalities in MRI scans, going beyond focal lesions to capture secondary structural changes like atrophy and ventricular enlargement.

[REFLECT](https://github.com/farzad-bz/REFLECT) learns an ordinary differential equation (ODE) that transports images from a pathological distribution to a healthy one. During inference, it generates a "healthy" counterfactual reconstruction of the input scan, allowing the system to extract an anomaly map based on the difference between the original input and the reconstruction.

## Features

* **Rectified Flow Inference:** Utilizes an Euler solver to transport latent representations along an ODE trajectory.
* **Hybrid Anomaly Maps:** Calculates final anomaly maps by blending pixel-space image differences with normalized latent-space differences.
* **Pre-Trained VAE Integration:** Automatically downloads required Medical-VAE checkpoints (`kl_f4` or `kl_f8`) directly from the Hugging Face Hub.
* **Advanced Evaluation Metrics:** Computes Free-Response Receiver Operating Characteristic (FROC) curves to evaluate sensitivity as a function of false positives per image (FPPI), which is particularly suited for scans with multiple structural anomalies.

## Prerequisites

Ensure you have the following Python packages installed to run the evaluation script:

* `torch`
* `torchvision`
* `numpy`
* `pandas`
* `scipy`
* `scikit-image`
* `matplotlib`
* `pyyaml`
* `huggingface_hub`
* `tqdm`

## Data Preparation

The script is configured to work with ATLAS and BraTS datasets.

* **Directory Structure:** Place your dataset inside a designated data folder (default is `Data/`). 
    
    *Example:*
    ```text
    Data/
    └── test/
        ├── sub-r001s001-slice_095-brainmask.png
        ├── sub-r001s001-slice_095-segmentation.png
        ├── sub-r001s001-slice_095-T1.png
        ├── sub-r001s002-slice_084-brainmask.png
        ├── sub-r001s002-slice_084-segmentation.png
        ├── sub-r001s002-slice_084-T1.png
        ├── sub-r001s003-slice_089-brainmask.png
        ├── sub-r001s003-slice_089-segmentation.png
        ├── sub-r001s003-slice_089-T1.png
        ├── sub-r001s006-slice_086-brainmask.png
        ├── sub-r001s006-slice_086-segmentation.png
        └── sub-r001s006-slice_086-T1.png
    ```

* **Ground Truth Annotations:** For FROC metric calculations on point-based expert clicks, ensure that the file `ClickLabels.csv` is located in your specified `--data-dir`.

## Model Configuration

The evaluation script expects a specific directory structure for your trained models. The parent directory of your `--model-path` must contain an `args.yml` file with the training configuration (generated when using the training scripts of REFLECT, see the [official repository](https://github.com/farzad-bz/REFLECT) to train your model).

## Usage

Run the evaluation script via the command line using `test_REFLECT.py`.

### Command Example

```bash
python test_REFLECT.py \
  --data-dir ./Data/ \
  --model-path ./checkpoints/my_model/model_weights.pt \
  --batch-size 1 \
  --backward-steps 5 \
  --threshold 0.5
```
### Arguments

* `--data-dir`: Path to the dataset and ground truth CSV (Default: Data/).
* `--model-path`: xplicit path to the trained UNet model .pt checkpoint (Required).
* `--batch-size`: Number of images to process per batch (Default: 1).
* `--backward-steps`: Number of ODE reverse steps for the Euler solver (Default: 5).
* `--threshold`: Binarization threshold used to generate segmentation maps for traditional metric calculation (Default: 0.5).
* `-num-workers`: Number of dataloader workers (Default: 4).

### Outputs

The script will perform inference over the specified test dataset and evaluate the generated anomaly maps against the ground truth. It will output:

    * A printed FROC Score (mean sensitivity across predefined FPPI thresholds: 0.25, 0.5, 1.0, 1.5), and Dice with selected thresholds.

    * A saved FROC curve plot (froc_curve.png) and a serialized Matplotlib figure (froc_curve.fig.pickle) located in the model's parent directory.

## Citation
```bibtex
@misc{mahé2025unsuperviseddetectionpoststrokebrain,
      title={Unsupervised Detection of Post-Stroke Brain Abnormalities}, 
      author={Youwan Mahé and Elise Bannier and Stéphanie Leplaideur and Elisa Fromont and Francesca Galassi},
      year={2025},
      eprint={2510.24398},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2510.24398}, 
}
