import argparse
import os

import numpy as np
import torch
from PIL import Image
from unidepth.models import UniDepthV1, UniDepthV2, UniDepthV2old
from unidepth.utils import eval_depth
from unidepth.utils.visualization import colorize, save_file_ply

scenes = {
    'Pat11_Step1': {'label_path': 'IHTest_202104_Path11_Step1_LWHSI1_collect0_DistStA_label.npz',
    'pca_path': 'IHTest_202104_Path11_Step1_LWHSI1_collect0_DistStA_pca.png'},
    'Path2_Step1': {'label_path': 'IHTest_202104_Path2_Step1_LWHSI1_collect0_DistStA_label.npz',
    'pca_path': 'IHTest_202104_Path2_Step1_LWHSI1_collect0_DistStA_pca.png'},
    'Path6_Step1': {'label_path':'IHTest_202104_Path6_Step1_LWHSI1_collect0_DistStA_label.npz',
    'pca_path': 'IHTest_202104_Path6_Step1_LWHSI1_collect0_DistStA_pca.png'},
    'Path19_Step1': {'label_path': 'IHTest_202104_Path19_Step1_LWHSI1_collect0_DistStA_label.npz',
    'pca_path': 'IHTest_202104_Path19_Step1_LWHSI1_collect0_DistStA_pca.png'},
    'Path29_Step1': {'label_path': 'IHTest_202104_Path29_Step1_LWHSI1_collect0_DistStA_label.npz',
    'pca_path': 'IHTest_202104_Path29_Step1_LWHSI1_collect0_DistStA_pca.png'},
    'Path4_Step22': {'label_path': 'IHTest_202108_Path4_Step22_LWHSI1_DistStA_label.npz',
    'pca_path': 'IHTest_202108_Path4_Step22_LWHSI1_DistStA_pca.png'},
    # 'Path11_Step1': {'heatcube_path': '/teamspace/uploads/ihdataset/IHTest_202104_DistStA/Path11_DistStA/Path11_Step1_DistStA/IHTest_202104_Path11_Step1_LWHSI1_collect0_DistStA_wavelength_corrected_m0p157_destriped_resampled.npz',
    # 'label_path': '/teamspace/uploads/ihdataset/IHTest_202104_DistStA/Path11_DistStA/Path11_Step1_DistStA/IHTest_202104_Path11_Step1_LWHSI1_collect0_DistStA_label.npz',
    # 'pca_path': '/teamspace/uploads/ihdataset/IHTest_202104_DistStA/Path11_DistStA/Path11_Step1_DistStA/IHTest_202104_Path11_Step1_LWHSI1_collect0_DistStA_pca.png'},
    # 'Path2_Step1':{'heatcube_path': '/teamspace/uploads/ihdataset/IHTest_202104_DistStA/Path2_DistStA/Path2_Step1_DistStA/IHTest_202104_Path2_Step1_LWHSI1_collect0_DistStA_wavelength_corrected_m0p157_destriped_resampled.npz', 
    # 'label_path': '/teamspace/uploads/ihdataset/IHTest_202104_DistStA/Path2_DistStA/Path2_Step1_DistStA/IHTest_202104_Path2_Step1_LWHSI1_collect0_DistStA_label.npz',
    # 'pca_path': '/teamspace/uploads/ihdataset/IHTest_202104_DistStA/Path2_DistStA/Path2_Step1_DistStA/IHTest_202104_Path2_Step1_LWHSI1_collect0_DistStA_pca.png'},
    # 'Path6_Step1': {'heatcube_path': '/teamspace/uploads/ihdataset/IHTest_202104_DistStA/Path6_DistStA/Path6_Step1_DistStA/IHTest_202104_Path6_Step1_LWHSI1_collect0_DistStA_wavelength_corrected_m0p157_destriped_resampled.npz', 
    # 'label_path':'/teamspace/uploads/ihdataset/IHTest_202104_DistStA/Path6_DistStA/Path6_Step1_DistStA/IHTest_202104_Path6_Step1_LWHSI1_collect0_DistStA_label.npz',
    # 'pca_path': '/teamspace/uploads/ihdataset/IHTest_202104_DistStA/Path6_DistStA/Path6_Step1_DistStA/IHTest_202104_Path6_Step1_LWHSI1_collect0_DistStA_pca.png'},
    # 'Path19_Step1': {'heatcube_path': '/teamspace/uploads/ihdataset/IHTest_202104_DistStA/Path19_DistStA/Path19_Step1_DistStA/IHTest_202104_Path19_Step1_LWHSI1_collect0_DistStA_wavelength_corrected_m0p157_destriped_resampled.npz', 
    # 'label_path': '/teamspace/uploads/ihdataset/IHTest_202104_DistStA/Path19_DistStA/Path19_Step1_DistStA/IHTest_202104_Path19_Step1_LWHSI1_collect0_DistStA_label.npz',
    # 'pca_path': '/teamspace/uploads/ihdataset/IHTest_202104_DistStA/Path19_DistStA/Path19_Step1_DistStA/IHTest_202104_Path19_Step1_LWHSI1_collect0_DistStA_pca.png'},
    # 'Path29_Step1': {'heatcube_path': '/teamspace/uploads/ihdataset/IHTest_202104_DistStA/Path29_DistStA/Path29_Step1_DistStA/IHTest_202104_Path29_Step1_LWHSI1_collect0_DistStA_wavelength_corrected_m0p157_destriped_resampled.npz', 
    # 'label_path': '/teamspace/uploads/ihdataset/IHTest_202104_DistStA/Path29_DistStA/Path29_Step1_DistStA/IHTest_202104_Path29_Step1_LWHSI1_collect0_DistStA_label.npz',
    # 'pca_path': '/teamspace/uploads/ihdataset/IHTest_202104_DistStA/Path29_DistStA/Path29_Step1_DistStA/IHTest_202104_Path29_Step1_LWHSI1_collect0_DistStA_pca.png'}, 
    # 'Path4_Step22': {'heatcube_path': '/teamspace/uploads/ihdataset/IHTest_202108_DistStA/Path4_DistStA/Path4_Step22/IHTest_202108_Path4_Step22_LWHSI1_DistStA_wavelength_corrected_m0p080_destriped_resampled.npz', 
    # 'label_path': '/teamspace/uploads/ihdataset/IHTest_202108_DistStA/Path4_DistStA/Path4_Step22/IHTest_202108_Path4_Step22_LWHSI1_DistStA_label.npz',
    # 'pca_path': '/teamspace/uploads/ihdataset/IHTest_202108_DistStA/Path4_DistStA/Path4_Step22/IHTest_202108_Path4_Step22_LWHSI1_DistStA_pca.png'},
    # 'Path27_Step9': {'heatcube_path': '/teamspace/uploads/data/heatcube_0002.npy', 
    # 'label_path': '/teamspace/uploads/data/step9_GTDepth.npy',
    # 'pca_path': 'figures/hadar_pca.png'}
}


def save(rgb, outputs, name, base_path, metrics, save_depth=False, 
        save_pointcloud=False, save_metric=False, show_colorbar=False,
        save_rays=False, max_depth=120.0):

    os.makedirs(base_path, exist_ok=True)

    depth = outputs["depth"]
    rays = outputs["rays"]
    points = outputs["points"]

    depth = depth.cpu().numpy()
    rays = ((rays + 1) * 127.5).clip(0, 255)
    if save_depth:
        if show_colorbar:
            import matplotlib.pyplot as plt
            from mpl_toolkits.axes_grid1 import make_axes_locatable

            fig, ax = plt.subplots(1, 1)
            img = ax.imshow(depth.squeeze(), cmap='plasma_r', vmin=0, vmax=max_depth)
            divider = make_axes_locatable(ax)
            # If you want a wider colorbar increase the size value
            cax = divider.append_axes("right", size='2%', pad='1%')
            ax.axis('off')
            fig.colorbar(img, cax=cax, orientation='vertical')
            cax.tick_params(labelsize=4) # To change the font size of colorbar labels
            plt.savefig(os.path.join(base_path, f"{name}_depth_colorbar.png"), 
                        bbox_inches='tight', dpi=300)
            plt.close()
        else:
            Image.fromarray(colorize(depth.squeeze(), cmap='plasma_r')).save(
                os.path.join(base_path, f"{name}_depth.png")
            )
    if save_rays:
        Image.fromarray(rays.squeeze().permute(1, 2, 0).byte().cpu().numpy()).save(
            os.path.join(base_path, f"{name}_rays.png")
        )

    if save_pointcloud:
        predictions_3d = points.permute(0, 2, 3, 1).reshape(-1, 3).cpu().numpy()
        rgb = rgb.permute(1, 2, 0).reshape(-1, 3).cpu().numpy()
        save_file_ply(predictions_3d, rgb, os.path.join(base_path, f"{name}.ply"))

    if save_metric:
        with open(os.path.join(base_path, f"{name}_metrics.txt"), 'w') as f:
            for key, value in metrics.items():
                f.write(f"{key}: {value:.4f}\n")

def eval_image(model, args):
    rgb = np.array(Image.open(args.input))
    rgb_torch = torch.from_numpy(rgb).permute(2, 0, 1)

    outputs = model.infer(rgb_torch)
    name = args.input.split("/")[-1].split(".")[0]

    depth = outputs["depth"].cpu().squeeze()

    gt_depth = torch.from_numpy(np.load(args.gt)['depth'])

    valid_mask = (torch.isnan(gt_depth) == 0) & (depth <= args.max_depth) & (depth >= 0.001)

    metrics = eval_depth(depth[valid_mask], gt_depth[valid_mask])

    save(
        rgb_torch,
        outputs,
        name=name,
        metrics=metrics,
        base_path=args.output,
        save_depth=args.save_depth,
        save_rays=args.save_rays,
        save_pointcloud=args.save_ply,
        save_metric=args.save_metric,
        show_colorbar=args.show_colorbar,
        max_depth=args.max_depth
    )

    return metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--input', type=str, required=True, help='Input image file path')
    # parser.add_argument('--gt', type=str, required=True, help='Ground truth depth file path (numpy .npy file)')
    parser.add_argument(
        "--output", type=str, required=False, help="Path to output directory."
    )
    parser.add_argument(
        "--config-file",
        type=str,
        required=True,
        default="./configs/eval/vitl.json",
        help="Path to config file. Please check ./configs/eval.",
    )
    parser.add_argument(
        "--save-depth", action="store_true", help="Save outputs as (colorized) png."
    )
    parser.add_argument(
        "--save-rays", action="store_true", help="Save ray visualization as png."
    )
    parser.add_argument(
        "--save-ply", action="store_true", help="Save pointcloud as ply."
    )
    parser.add_argument(
        "--save-metric", action="store_true", help="Save evaluation metrics to a text file."
    )
    parser.add_argument(
        "--show-colorbar", action="store_true", help="Show colorbar in depth map."
    )
    parser.add_argument(
        "--max-depth", type=float, default=120.0, help="Maximum depth value for evaluation."
    )
    parser.add_argument(
        "--model-type", type=str, default="unidepth-v1", 
        help="Type of the model to use.", choices=["unidepth-v1", "unidepth-v2", "unidepth-v2old"]
    )
    parser.add_argument(
        "--data-root", type=str, default="", help="Root directory for the dataset."
    )
    args = parser.parse_args()

    print("Torch version:", torch.__version__)
    version = args.config_file.split("/")[-1].split(".")[0].split("_")[-1]
    name = f"{args.model_type}-{version}"

    if args.model_type == "unidepth-v1":
        model = UniDepthV1.from_pretrained(f"lpiccinelli/{name}")
    elif args.model_type == "unidepth-v2":
        model = UniDepthV2.from_pretrained(f"lpiccinelli/{name}")
    elif args.model_type == "unidepth-v2old":
        model = UniDepthV2old.from_pretrained(f"lpiccinelli/{name}")
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    mean_metrics = { 'abs_rel': 0.0, 'sq_rel': 0.0, 'rmse': 0.0, 
                    'rmse_log': 0.0, 'd1': 0.0, 'd2': 0.0, 'd3': 0.0,
                    'log10': 0.0, 'silog': 0.0 }

    if args.output is None:
        args.output = f'outputs_{args.model_type.lower()}'

    for scene in scenes.values():
        args.input = os.path.join(args.data_root, scene['pca_path'])
        args.gt = os.path.join(args.data_root, scene['label_path'])
        metrics = eval_image(model, args)
        for key in mean_metrics.keys():
            mean_metrics[key] += metrics[key]
    num_scenes = len(scenes)
    print("Average metrics over all scenes:")
    for key in mean_metrics.keys():
        mean_metrics[key] /= num_scenes
        print(f"{key}: {mean_metrics[key]:.4f}")
        
    # Save average metrics to a file
    with open(os.path.join(args.output, "average_metrics.txt"), 'w') as f:
        for key, value in mean_metrics.items():
            f.write(f"{key}: {value:.4f}\n")
