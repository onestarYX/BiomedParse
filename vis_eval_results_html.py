import os, sys
from pathlib import Path
import pickle
from ..html_tools.html_utils import *
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde


if __name__ == "__main__":
    output_dir = Path('eval_output')
    html_path = output_dir / 'eval_results.html'
    html_assets_dir = output_dir / 'html_assets'
    html_assets_dir.mkdir(parents=True, exist_ok=True)
    results_input_dir = Path('eval_output/pannuke123_resolveMask')
    stats_path = results_input_dir / 'stats.pkl'

    with open(stats_path, "rb") as pickle_file:
        _ = pickle.load(pickle_file)
        img_dice_dict = pickle.load(pickle_file)

    class_names = [
        "Neoplastic",   # 0
        "Inflammatory", # 1
        "Connective",   # 2
        "Dead",         # 3
        "Epithelial"    # 4
    ]
    keys = np.array(list(img_dice_dict.keys()))
    class_dice_dict = {}
    for i in range(len(class_names)):
        cur_dice_list = []
        for k in keys:
            cur_dice_list.append(img_dice_dict[k][i])

        class_dice_dict[class_names[i]] = np.array(cur_dice_list)

    # Plot stats distribution
    for class_name in class_names:
        arr = class_dice_dict[class_name]
        plt.figure(figsize=(5, 5))
        plt.hist(arr, bins=30, density=True, alpha=0.6, edgecolor='gray', label='hist')
        kde = gaussian_kde(arr)
        xs = np.linspace(arr.min(), arr.max(), 200)
        plt.plot(xs, kde(xs), lw=2, label='KDE')
        plt.xlabel("Value")
        plt.ylabel("Density")
        plt.title(f"Distribution of {class_name}: mean={np.mean(arr)}, std={np.std(arr)}")
        plt.legend()
        plt.tight_layout()
        fig_path = html_assets_dir / f'{class_name}_hist.png'
        plt.savefig(fig_path)


    with open(html_path, "w", encoding="utf-8") as f:
        write_start(f)
        write_head(f, level=1, msg='BiomedParse Semantic Segmentation on Pannuke')

        write_head(f, level=2, msg='Class Dice Score Distribution')
        for class_name in class_names:
            fig_path = html_assets_dir / f'{class_name}_hist.png'
            write_img(f, img_path=fig_path.resolve(), caption='')

        num_imgs = 3
        for class_name in class_names:
            write_head(f, level=2, msg=f'{class_name} Examples')

            arr = class_dice_dict[class_name]
            valid_mask = arr != 1
            arr_masked = arr[valid_mask]
            keys_masked = keys[valid_mask]

            write_head(f, level=3, msg='High Dice Score')
            indices_h_to_l = np.argsort(arr_masked)[::-1]
            for k in keys_masked[indices_h_to_l][:num_imgs]:
                img_path = results_input_dir / f'{k}.png'
                write_img(f, img_path.resolve(), caption='')

            write_head(f, level=3, msg='Low Dice Score')
            indices_l_to_h = indices_h_to_l[::-1]
            for k in keys_masked[indices_l_to_h][:num_imgs]:
                img_path = results_input_dir / f'{k}.png'
                write_img(f, img_path.resolve(), caption='')

            write_head(f, level=3, msg='Random examples with Dice Score in [0.3, 0.6]')
            indices_mid_range = np.where((arr >= 0.3) & (arr <= 0.6))
            for k in keys[indices_mid_range][:num_imgs]:
                img_path = results_input_dir / f'{k}.png'
                write_img(f, img_path.resolve(), caption='')

        write_end(f)