import argparse
import os
import pickle

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from huggingface_hub import hf_hub_download
from scipy.ndimage import gaussian_filter, label
from skimage.transform import resize
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from medical_models import UNET_models
from MedicalDataLoader import ATLASDataset, BraTS2021Dataset
from metrics.froc import (
    compute_fp_tp_probs_componentwise,
    compute_froc_curve_data,
    compute_froc_score,
)


def smooth_mask(mask, sigma=1.0):
    smoothed_mask = gaussian_filter(mask, sigma=sigma)
    return smoothed_mask


def compute_dice(pred_mask, gt_mask):
    """
    Computes the Dice coefficient between a binary prediction and ground truth.
    Returns 1.0 if both masks are empty.
    """
    pred_mask = np.asarray(pred_mask).astype(bool)
    gt_mask = np.asarray(gt_mask).astype(bool)

    intersection = np.logical_and(pred_mask, gt_mask).sum()
    sum_masks = pred_mask.sum() + gt_mask.sum()

    if sum_masks == 0:
        return 1.0
    
    return 2.0 * intersection / sum_masks


def compute_froc_monai(
    anomaly_maps,
    anoGT,
    segmentations,
    filenames,
    eval_thresholds=(0.25, 0.5, 1, 1.5),
    save_dir=None,
    click_radius=5,
    prob_threshold=0.036016,
):
    """
    Compute FROC curve and AUFROC using MONAI-like logic but component-based.

    - Ground-truth evaluation masks are 5x5 (radius=2) around click coordinates.
    - Predictions are obtained from connected components above a given probability threshold.
    - Each component is assigned its max probability as detection score.
    """
    n_images = len(filenames)
    fp_probs, tp_probs = [], []
    total_targets = 0

    for amap, fname in zip(anomaly_maps, filenames):
        # --- Step 1. Build evaluation mask from GT clicks (5x5 squares) ---
        gt_coords = anoGT[anoGT["SubjectID"] == fname][["y", "x"]].values
        eval_mask = np.zeros_like(amap, dtype=np.uint8)

        for y, x in gt_coords:
            y, x = int(round(y)), int(round(x))
            y1, y2 = max(0, y - click_radius), min(amap.shape[0], y + click_radius + 1)
            x1, x2 = max(0, x - click_radius), min(amap.shape[1], x + click_radius + 1)
            eval_mask[y1:y2, x1:x2] = 1

        # Skip images with no GT
        if eval_mask.sum() == 0:
            continue

        # --- Step 2. Threshold anomaly map to create binary mask ---
        bin_map = (amap >= prob_threshold).astype(np.uint8)
        labeled_map, num_components = label(bin_map)

        if num_components == 0:
            continue

        # --- Step 3. Extract component masks and probabilities ---
        probs = []
        component_masks = []
        for comp_id in range(1, num_components + 1):
            comp_mask = (labeled_map == comp_id)
            if np.any(comp_mask):
                component_masks.append(comp_mask)
                probs.append(float(amap[comp_mask].max()))

        # --- Step 4. Compute FP and TP probabilities (component-based) ---
        fpp, tpp, num_targets = compute_fp_tp_probs_componentwise(
            probs=probs,
            component_masks=component_masks,
            evaluation_mask=eval_mask,
            labels_to_exclude=None
        )

        fp_probs.extend(fpp)
        tp_probs.extend(tpp)
        total_targets += num_targets

    # --- Step 5. Check if valid detections exist ---
    if total_targets == 0 or (len(tp_probs) == 0 and len(fp_probs) == 0):
        print("⚠️ No valid detections or ground-truth targets found — returning empty results.")
        return {"fps_per_image": [], "total_sensitivity": [], "froc_score": 0.0}

    fp_probs = np.array(fp_probs)
    tp_probs = np.array(tp_probs)

    # --- Step 6. Compute FROC curve data ---
    fps_per_image, total_sensitivity = compute_froc_curve_data(
        fp_probs, tp_probs, total_targets, num_images=n_images
    )

    # --- Step 7. Compute FROC score ---
    froc_score = compute_froc_score(fps_per_image, total_sensitivity, eval_thresholds)

    # --- Step 8. Plot FROC curve ---
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fps_per_image, total_sensitivity, marker="o", label="FROC")
    ax.set_xlabel("False Positives per Image (FPPI)")
    ax.set_ylabel("Sensitivity")
    ax.set_title(f"FROC Curve (thr={prob_threshold}, 5x5 GT Region)")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        # Save as static image
        fig.savefig(os.path.join(save_dir, "froc_curve.png"))
        # Save as serialized figure
        with open(os.path.join(save_dir, "froc_curve.fig.pickle"), "wb") as f:
            pickle.dump(fig, f)

    plt.close(fig)

    print(f"✅ FROC Score (mean sensitivity @ {eval_thresholds}): {froc_score:.4f}")

    return {
        "fps_per_image": fps_per_image,
        "total_sensitivity": total_sensitivity,
        "froc_score": froc_score,
    }


def calculate_metrics(ground_truth, prediction, filenames, threshold):
    print("\n--- Computing Evaluation Metrics ---")
    
    # 1. Compute Dice Scores
    # Binarize predictions based on the user-defined threshold
    binary_prediction = (prediction >= threshold).astype(np.int32)
    
    # Global Dice (all pixels across all evaluated slices)
    global_dice = compute_dice(binary_prediction, ground_truth)
    print(f"✅ Global Dice Score (threshold={threshold}): {global_dice:.4f}")

    # 2. Anomaly Detection score (FROC)
    gt_path = os.path.join(args.data_dir, "merged_anoGT_Class1.csv")
    if os.path.exists(gt_path):
        anoGT = pd.read_csv(gt_path)
        compute_froc_monai(prediction, anoGT, ground_truth, filenames, eval_thresholds=[0.25, 0.5, 1, 1.5], save_dir=args.parent_dir)
    else:
        print(f"⚠️ FROC GT not found at {gt_path}. Skipping FROC evaluation.")
        
    return global_dice


def evaluate(x0s, segmentations, encodeds, image_samples, latent_samples, filenames, args):
    anomaly_maps = []
    gt = []

    for x, segmentation, encoded, image_sample, latent_sample in zip(x0s, segmentations, encodeds, image_samples, latent_samples):
        image_difference = (((((torch.abs(image_sample-x))).to(torch.float32)).mean(axis=0)).detach().cpu().numpy().transpose(1, 2, 0).max(axis=2))
        image_difference = (np.clip(image_difference, 0.0, 0.4)) * 2.5
        image_difference = smooth_mask(image_difference, sigma=3)
        
        latent_difference = (((((torch.abs(latent_sample-encoded))).to(torch.float32)).mean(axis=0)).detach().cpu().numpy().transpose(1, 2, 0).mean(axis=2))
        latent_difference = (np.clip(latent_difference, 0.0, 0.4)) * 2.5
        latent_difference = smooth_mask(latent_difference, sigma=1)
        latent_difference = resize(latent_difference, (args.image_size, args.image_size))
        
        final_anomaly = 1/2 * image_difference + 1/2 * latent_difference
        anomaly_maps.append(final_anomaly)
        gt.append(segmentation[0, :, :].cpu().numpy())
            
    anomaly_maps = np.stack(anomaly_maps, axis=0)
    
    gt = np.stack(gt, axis=0)
    gt = (gt > 0).astype(np.int32)

    calculate_metrics(gt, anomaly_maps, filenames, args.threshold)

    return


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
     
    if args.vae == 'kl_f4':
        in_channels = out_channels = 3
    else:
        in_channels = out_channels = 4
        
    model = UNET_models[args.model](in_channels=in_channels, out_channels=out_channels)
    try:
        state_dict = torch.load(args.model_path, weights_only=False)['model']
        print(model.load_state_dict(state_dict))
    except:
        raise Exception('Provided trained model path could not be found or it is not consistent with model params.')
    
    model.eval()
    model.to(device)    

    if args.vae == 'kl_f4':
        vae_model_path = hf_hub_download(repo_id="farzadbz/Medical-VAE", filename="VAE-Medical-klf4.pt")
        vae = torch.load(vae_model_path, weights_only=False)
        embedding_dim = 3
        compression_factor = 4
        
    elif args.vae == 'kl_f8':
        vae_model_path = hf_hub_download(repo_id="farzadbz/Medical-VAE", filename="VAE-Medical-klf8.pt")
        vae = torch.load(vae_model_path, weights_only=False)
        embedding_dim = 4
        compression_factor = 8
        
    vae.eval()
    vae.to(device)
    
    # Setup data:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5], inplace=True)
    ])
    
    if args.dataset == 'BraTS':
        test_dataset = BraTS2021Dataset('test', rootdir=args.data_dir, transform=transform, image_size=args.image_size, augment=False, modality=args.modality, embedding_dim=embedding_dim, compression_factor=compression_factor)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=False)
    else:
        test_dataset = ATLASDataset('test', rootdir=args.data_dir, transform=transform, image_size=args.image_size, augment=False, modality=args.modality, embedding_dim=embedding_dim, compression_factor=compression_factor)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=False)
    
    print(f"Dataset contains {len(test_dataset)} Test images.")

    x_s = []
    encoded_s = []
    image_samples_s = []
    latent_samples_s = []
    x0_s = []
    segmentation_s = []
    filenames = []
    
    print('=-='*20)
    print('Starting evaluation...')
    print('=-='*20)
    for ii, (x, mask, seg, fname) in enumerate(tqdm(test_loader, desc="Inference", unit="it")):
        with torch.no_grad():
            # Map input images to latent space + normalize latents:
            encoded = vae.encode(x.to(device)).mean.mul_(0.18215) # Normalization params got from LDM package
            
            latent_sample = encoded.clone()
            
            # Euler solver (can use higher-order methods)
            dt = 1 / args.backward_steps   # Step size (adjust for accuracy/speed tradeoff)
            for time in torch.arange(0, 1, dt):
                t = time * torch.ones((encoded.shape[0], 1)).to(torch.float32).to(device)
                velocity = model(latent_sample, t)
                latent_sample = latent_sample + velocity * dt

            image_samples = vae.decode(latent_sample / 0.18215)
            x0 = vae.decode(encoded / 0.18215)

            x_s += [_x.unsqueeze(0) for _x in x]
            segmentation_s += [_seg.unsqueeze(0) for _seg in seg]
            encoded_s += [_encoded.unsqueeze(0) for _encoded in encoded]
            image_samples_s += [_image_samples.unsqueeze(0) for _image_samples in image_samples]
            latent_samples_s += [_latent_samples.unsqueeze(0) for _latent_samples in latent_sample]
            x0_s += [_x0.unsqueeze(0) for _x0 in x0]
            filenames += [os.path.splitext(os.path.basename(f))[0] for f in fname]

    evaluate(x0_s, segmentation_s, encoded_s, image_samples_s, latent_samples_s, filenames, args)
    print('=-='*20)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--data-dir", type=str, default='Data/')
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--model-path", type=str)
    parser.add_argument("--backward-steps", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.5)
    
    args = parser.parse_args()
    
    args.parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(args.model_path)))
    try:
        with open(os.path.join(args.parent_dir, 'args.yml'), 'r') as file:
            config = yaml.safe_load(file)  
        args.dataset = config['dataset']
        args.image_size = int(config['image_size'])
        args.modality = config['modality']
        args.vae = config['vae']
        args.model = config['model']          
    except:
        raise Exception("YAML config file could not be found in the parent folder of the provided model path")

    main(args)