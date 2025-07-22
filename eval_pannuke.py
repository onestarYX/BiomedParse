from PIL import Image
import os, sys

# Get the absolute path to the current file
current_file = os.path.abspath(__file__)
grandparent_dir = os.path.dirname(os.path.dirname(current_file))
sys.path.insert(0, grandparent_dir)

import argparse
import torch
import cv2
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
import prettytable as pt
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pickle

from seg_dataset import DataFolder, test_collate_fn

from BiomedParse.modeling.BaseModel import BaseModel
from BiomedParse.modeling import build_model
from BiomedParse.utilities.distributed import init_distributed
from BiomedParse.utilities.arguments import load_opt_from_config_files
from BiomedParse.utilities.constants import BIOMED_CLASSES

from BiomedParse.inference_utils.inference import interactive_infer_image
from BiomedParse.inference_utils.output_processing import check_mask_stats, combine_masks


def dice_score(mask1: np.ndarray, mask2: np.ndarray, eps: float = 1e-6) -> float:
    """
    Compute Dice coefficient between two binary masks.

    Parameters
    ----------
    mask1, mask2 : np.ndarray
        Binary arrays (0/1) of the same shape.
    eps : float
        Small constant to avoid division by zero when both masks are empty.

    Returns
    -------
    float
        Dice coefficient.
    """
    # ensure boolean
    m1 = mask1.astype(bool)
    m2 = mask2.astype(bool)

    # intersection and sums
    intersection = np.logical_and(m1, m2).sum()
    sum_sizes = m1.sum() + m2.sum()

    # Dice formula
    return (2.0 * intersection + eps) / (sum_sizes + eps)

@torch.inference_mode()
def evaluate(
        cfg,
        model,
        test_loader,
        device,
        epoch=0,
        output_dir=None
):
    model.eval()

    class_names = [
        "Neoplastic",   # 0
        "Inflammatory", # 1
        "Connective",   # 2
        "Dead",         # 3
        "Epithelial"    # 4
    ]
    # np.random.seed(19)
    # class_colors = (np.random.rand(len(class_names), 3) * 255).astype(np.uint8)
    class_colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [0, 255, 255]])
    if "use_know_kw" in cfg and cfg.use_known_kw:
        text_prompts = ['neoplastic cells', 'inflammatory cells', 'connective tissue cells']
    else:
        text_prompts = ['neoplastic cells', 'inflammatory cells', 'connective tissue cells', 'dead cells',
                       'epithelial cells']

    def get_dice_img_caption(dice_dict):
        msg = "DICE Scores: "
        for i in range(len(class_names)):
            msg += f"{class_names[i]}: {dice_dict[i]:.4f}"
            if i != len(class_names) - 1:
                msg += ", "
        return msg

    epoch_iterator = tqdm(test_loader, file=sys.stdout, desc="Test (X / X Steps)",
                          dynamic_ncols=True)
    class_dice_scores = {}
    for class_name in class_names:
        class_dice_scores[class_name] = []
    image_results = {}
    for data_iter_step, (images, inst_maps, type_maps, prompt_points, prompt_labels, prompt_cell_types,
                         cell_nums, ori_sizes, file_inds) in enumerate(epoch_iterator):

        assert len(images) == 1, 'batch size must be 1'

        epoch_iterator.set_description(
            "Epoch=%d: Test (%d / %d Steps) " % (epoch, data_iter_step, len(test_loader)))

        image = images[0].permute(1, 2, 0).numpy()

        pred_mask = interactive_infer_image(model, Image.fromarray(image), text_prompts)
        prompt_mask = {k: pred_mask[i] for i, k in enumerate(text_prompts)}
        resolved_mask = combine_masks(prompt_mask)

        new_mask = np.zeros_like(pred_mask).astype(np.uint8)
        for k, mask in resolved_mask.items():
            idx = text_prompts.index(k)
            new_mask[idx] = mask
        pred_mask = new_mask

        # Resize mask to original shape
        pred_mask_resized = np.zeros((pred_mask.shape[0], 256, 256), dtype=pred_mask.dtype)
        for i in range(pred_mask.shape[0]):
            pred_mask_resized[i] = cv2.resize(pred_mask[i], (256, 256), interpolation=cv2.INTER_LINEAR)
        confidence_threshold = 0.5
        pred_mask_resized = pred_mask_resized > confidence_threshold

        # Compute DICE score and store segmented outputs
        file_path = Path(test_loader.dataset.files[file_inds])
        gt_type_map = np.zeros((256, 256, 3), dtype=np.uint8)
        # pred_type_map = np.zeros((256, 256, 3), dtype=np.uint8)
        pred_type_maps = [np.zeros((256, 256, 3), dtype=np.uint8) for _ in range(len(class_names))]
        image_resized = cv2.resize(image, (256, 256), interpolation=cv2.INTER_LINEAR)
        for i in range(len(class_names)):
            if "use_known_kw" in cfg and cfg.use_known_kw:
               if i >= 3:
                   break
            gt_type_mask = (type_maps.squeeze(0).cpu().numpy() == (i+1))
            pred_type_mask = pred_mask_resized[i]
            cur_dice_score = dice_score(pred_type_mask, gt_type_mask)
            class_dice_scores[class_names[i]].append(cur_dice_score)
            if file_path.stem not in image_results:
                image_results[file_path.stem] = {i: cur_dice_score}
            else:
                image_results[file_path.stem][i] = cur_dice_score

            # Save segmentation images
            gt_type_map[gt_type_mask] = class_colors[i]
            pred_type_maps[i][pred_type_mask] = class_colors[i]


        gt_type_map = (gt_type_map.astype(float) * 0.5 + image_resized.astype(float) * 0.5).astype(np.uint8)
        # pred_type_map = (pred_type_map.astype(float) * 0.5 + image_resized.astype(float) * 0.5).astype(np.uint8)
        for i in range(len(class_names)):
            pred_type_maps[i] = (pred_type_maps[i].astype(float) * 0.5 + image_resized.astype(float) * 0.5).astype(np.uint8)

        image_results[file_path.stem]['image'] = image_resized
        image_results[file_path.stem]['gt_type_map'] = gt_type_map
        image_results[file_path.stem]['pred_type_maps'] = pred_type_maps

        fig, ax = plt.subplots(1, 2 + len(class_names), figsize=(15, 3))
        ax[0].imshow(image_resized); ax[0].axis('off'); ax[0].set_title('Original Image')
        ax[1].imshow(gt_type_map); ax[1].axis('off'); ax[1].set_title('Ground Truth')

        legend_handles = []
        for i in range(len(class_names)):
            ax[i + 2].imshow(pred_type_maps[i]); ax[i + 2].axis('off')
            ax[i + 2].set_title(f"{class_names[i]}: {image_results[file_path.stem][i]:.3f}")
            legend_handles.append(Patch(facecolor=(class_colors[i] / 255.0).tolist(), edgecolor='k', label=class_names[i]))

        fig.legend(handles=legend_handles, title="Classes")

        # caption = get_dice_img_caption(image_results[file_path.stem])
        # fig.suptitle(caption)
        # plt.tight_layout()
        plt.savefig(output_dir / file_path.name)
        plt.close(fig)


    class_dice_scores_mean = []
    for class_name in class_names:
        class_dice_scores_mean.append(np.mean(class_dice_scores[class_name]))

    # Save stats
    with open(output_dir / 'stats.pkl', 'wb') as f:
        pickle.dump(class_dice_scores, f)
        pickle.dump(image_results, f)


    # Print table
    table = pt.PrettyTable()
    table.add_column('CLASS', class_names)

    table.add_column('DICE Scores', np.array(class_dice_scores_mean).round(2))

    table.add_row(['---'] * 2)

    table.add_row(['Avg', np.mean(class_dice_scores_mean).round(2)])
    print(table)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str,
                        default="BiomedParse/checkpoints/biomedparse_v1.pt")
    parser.add_argument('--config', type=str,
                        default="configs/promptnucseg_pannuke123_seg.yaml")
    args, extras = parser.parse_known_args()

    ckpt_path = Path(args.ckpt)
    assert (ckpt_path.exists())

    config_path = args.config
    if config_path is None:
        config_path = ckpt_path.parent / 'config.yaml'
    assert (os.path.exists(config_path))
    base_cfg = OmegaConf.load(config_path)
    cli_cfg = OmegaConf.from_cli(extras)
    cfg = OmegaConf.merge(base_cfg, cli_cfg)

    # Load model
    opt = load_opt_from_config_files(["BiomedParse/configs/biomedparse_inference.yaml"])
    opt = init_distributed(opt)

    pretrained_pth = args.ckpt
    model = BaseModel(opt, build_model(opt)).from_pretrained(pretrained_pth).eval().cuda()
    with torch.no_grad():
        model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(BIOMED_CLASSES + ["background"],
                                                                            is_eval=True)

    # Load dataset
    transform = [
        dict(type='Resize', height=1024, width=1024),
        # dict(type='Normalize'),
    ]
    test_dataset = DataFolder(cfg, 'test', transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        sampler=None,
        num_workers=8,
        collate_fn=test_collate_fn
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path('BiomedParse/eval_output/pannuke123_resolveMask')
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluate(
        cfg,
        model,
        test_loader,
        device,
        output_dir=output_dir
    )