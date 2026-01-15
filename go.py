import argparse, json, os
from glob import glob
import shutil
import numpy as np
from PIL import Image as PImage
import torch
import warnings

# Custom modules
from functions import compute_ICE, ants_affine_registration

def load_and_resize(data_path, img_size=(256, 256)):
    """
    Load fixed/moving images and optional segmentation masks, resize to img_size,
    normalize intensity to [0,1], and compute spacing_ratio using ORIGINAL image sizes
    (before resize) and optional pixel spacing.

    Required keys in data_path:
        - "fixed": path to fixed image
        - "moving": path to moving image

    Optional keys:
        - "fixed_seg": path to fixed segmentation mask (binary 0/255)
        - "moving_seg": path to moving segmentation mask (binary 0/255)
        - "fixed_pixel_spacing": [sx, sy] pixel spacing of fixed image (default [1,1])
        - "moving_pixel_spacing": [sx, sy] pixel spacing of moving image (default [1,1])

    Returns:
        im_f: (H,W) float32 in [0,1]
        im_m: (H,W) float32 in [0,1]
        seg_f: (H,W) float32 in {0,1} or None
        seg_m: (H,W) float32 in {0,1} or None
        spacing_ratio: float
            (fixed physical area) / (moving physical area)
            = (Hf*Wf*sfx*sfy) / (Hm*Wm*smx*smy)
    """
    # ---------- check required paths ----------
    fixed_path = data_path.get("fixed", None)
    moving_path = data_path.get("moving", None)
    if fixed_path is None or not os.path.exists(fixed_path):
        raise FileNotFoundError("fixed image path does not exist")
    if moving_path is None or not os.path.exists(moving_path):
        raise FileNotFoundError("moving image path does not exist")

    # ---------- load images (keep original size for spacing_ratio) ----------
    pil_f_orig = PImage.open(fixed_path)
    pil_m_orig = PImage.open(moving_path)

    # PIL size is (W, H)
    Wf0, Hf0 = pil_f_orig.size
    Wm0, Hm0 = pil_m_orig.size

    # Convert to grayscale and resize for computation
    pil_f = pil_f_orig.convert("L").resize(img_size, resample=PImage.BILINEAR)
    pil_m = pil_m_orig.convert("L").resize(img_size, resample=PImage.BILINEAR)

    im_f = np.asarray(pil_f, dtype=np.float32)
    im_m = np.asarray(pil_m, dtype=np.float32)

    # ---------- normalize to [0,1] ----------
    im_f = (im_f - im_f.min()) / (im_f.max() - im_f.min() + 1e-8)
    im_m = (im_m - im_m.min()) / (im_m.max() - im_m.min() + 1e-8)

    # ---------- optional segmentation (fixed) ----------
    seg_f = None
    fixed_seg_path = data_path.get("fixed_seg", None)
    if fixed_seg_path is not None and os.path.exists(fixed_seg_path):
        seg_f_pil = PImage.open(fixed_seg_path).convert("L").resize(
            img_size, resample=PImage.NEAREST
        )
        seg_f = np.asarray(seg_f_pil, dtype=np.float32)
        seg_f = (seg_f == 255).astype(np.float32)

    # ---------- optional segmentation (moving) ----------
    seg_m = None
    moving_seg_path = data_path.get("moving_seg", None)
    if moving_seg_path is not None and os.path.exists(moving_seg_path):
        seg_m_pil = PImage.open(moving_seg_path).convert("L").resize(
            img_size, resample=PImage.NEAREST
        )
        seg_m = np.asarray(seg_m_pil, dtype=np.float32)
        seg_m = (seg_m == 255).astype(np.float32)

    # ---------- spacing ratio using ORIGINAL sizes ----------
    fixed_spacing = data_path.get("fixed_pixel_spacing", None)
    moving_spacing = data_path.get("moving_pixel_spacing", None)
    
    spacing_ratio = None

    if fixed_spacing is not None and moving_spacing is not None:
        fixed_spacing = np.asarray(fixed_spacing, dtype=np.float64)
        moving_spacing = np.asarray(moving_spacing, dtype=np.float64)
        
        fixed_area_phys = (Hf0 * Wf0) * (fixed_spacing[0] * fixed_spacing[1])
        moving_area_phys = (Hm0 * Wm0) * (moving_spacing[0] * moving_spacing[1])

        spacing_ratio = float(fixed_area_phys / (moving_area_phys + 1e-12))

    return im_f, im_m, seg_f, seg_m, spacing_ratio



def print_stats(name, values):
    """
    Print mean and standard deviation of a metric list.
    """
    arr = np.array(values, dtype=np.float64)
    print(f"{name}: mean = {arr.mean():.6f}, std = {arr.std(ddof=0):.6f}", end=" ")
    
def summarize_metrics(args, rm_results=False):
    """
    Aggregate metrics from all registration results and print statistics.

    Args:
        args: command line arguments containing result_path
        rm_results: if True, remove the result directory after summarizing
    """
    # Find all metrics.json files in result_path
    paths = glob(os.path.join(args.result_path, "*", "metrics.json"))
    
    # Initialize lists for each metric
    SSIM_list, NMI_list, ICE_list = [], [], []
    TV_mean_list, TV_p95_list = [], []
    DJ_fold_list, DJ_p95_list = [], []
    DJ_p5_list = []
    MSE_DJ_c_list = [], []


    # Load each metrics.json and append values to lists
    for path in paths:
        with open(path, 'r') as f:
            data = json.load(f)
            SSIM_val = data.get("SSIM")
            NMI_val = data.get("NMI")
            ice_val = data.get("ICE")
            TV_mean = data.get("TV_mean")
            TV_p95 = data.get("TV_p95")
            DJ_fold = data.get("DJ_fold%")
            DJ_p95 = data.get("DJ_p95")
            DJ_p5 = data.get("DJ_p5")
            MSE_DJ_c = data.get("MSE[D_J-c]")

        SSIM_list.append(SSIM_val); NMI_list.append(NMI_val); ICE_list.append(ice_val)
        TV_mean_list.append(TV_mean); TV_p95_list.append(TV_p95)
        DJ_fold_list.append(DJ_fold); DJ_p95_list.append(DJ_p95); DJ_p5_list.append(DJ_p5)
        MSE_DJ_c_list.append(MSE_DJ_c)
        
    # Print summary statistics
    if args.LAP_type == "original":
        print(f"LAP_type:{args.LAP_type}, n={len(SSIM_list)}")
    else:
        print(f"LAP_type:{args.LAP_type}, beta={args.beta}, gamma={args.gamma}, N={args.number_of_F_basis} , n={len(SSIM_list)}")
        
    print_stats("SSIM", SSIM_list)
    print_stats("NMI", NMI_list)
    print_stats("ICE", ICE_list)
    print_stats("MSE[D_J-c]", MSE_DJ_c_list)
    print("")
    print_stats("TV_mean", TV_mean_list)
    print_stats("TV_p95", TV_p95_list)
    print_stats("DJ_fold%", DJ_fold_list)
    print_stats("DJ_p95", DJ_p95_list)
    print_stats("DJ_p5", DJ_p5_list)
    print("")
    
    # Optionally remove results directory for next run
    if rm_results:
        shutil.rmtree(args.result_path, ignore_errors=True)

def main():
    
    backward = True  # Set to False to skip ICE computation
    
    # Parse arguments     
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_paths_json", default='test_data/test.json')
    parser.add_argument("--img_size", default=[256, 256], type=int, nargs=2)
    parser.add_argument("--LAP_type", default="proposed", choices=["proposed", "original"])
    parser.add_argument("--result_path", default=None)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--sigma", default="proposed", choices=["proposed", "original"])
    parser.add_argument("--r_list", default=[8, 8, 8, 4, 4, 2, 2, 1, 1], type=int, nargs='+')
    
    # The following parameters are used only when LAP_type is "proposed"
    parser.add_argument("--number_of_F_basis", type=int, default=5)
    parser.add_argument("--beta", default=[5, 150000], type=float, nargs=2)
    parser.add_argument("--gamma", type=float, default=0)
    args = parser.parse_args()
    
    # Validate arguments
    assert os.path.exists(args.data_paths_json), "data_paths_json does not exist"

    # Import LAP class based on selected type
    if args.LAP_type == "proposed":
        from lap_proposed import LAP
    elif args.LAP_type == "original":
        from lap_original import LAP
    else:
        print("Unrecognized LAP_type, use proposed instead.")
        from lap_proposed import LAP
    
    # Select device
    device = args.device
    if device in ("auto", "cuda"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"====== Running on {device} ======")
    args.device = device
    
    # Set default result path if not provided
    if args.result_path is None:
        if args.LAP_type == "proposed":
            args.result_path = "TBRLAP_results"
        elif args.LAP_type == "original":
            args.result_path = "LAP_results"
    
    # Create result directory
    if os.path.exists(args.result_path):
        warnings.warn(f"result_path {args.result_path} already exists, results may be overwritten")
    os.makedirs(args.result_path, exist_ok=True)
    
    # Load dataset JSON
    with open(args.data_paths_json, "r", encoding="utf-8") as f:
        data_paths = json.load(f)
        
    # Loop over each image pair
    for i in range(len(data_paths)):
        
        # Load and preprocess images/masks
        im_f, im_m, seg_f, seg_m, spacing_ratio = load_and_resize(data_paths[i], args.img_size)
        
        initial_field = ants_affine_registration(im_f, im_m)
        # Forward registration
        reg = LAP(
                fixed_img=im_f,
                moving_img=im_m,
                initial_field=initial_field,
                args=args,
                conserve_region=seg_m,
                spacing_ratio=spacing_ratio
                )
        reg.run_iterations()
        
        if backward:
            
            if spacing_ratio is not None:
                spacing_ratio = 1 / spacing_ratio
                
            initial_field = ants_affine_registration(im_m, im_f)
            # Backward registration
            bwd_reg = LAP(
                    fixed_img=im_m,
                    moving_img=im_f,
                    initial_field=initial_field,
                    args=args,
                    conserve_region=seg_f,
                    spacing_ratio=spacing_ratio
                    )
            bwd_reg.run_iterations()
        
            # Compute inverse-consistency error (ICE)
            ICE = compute_ICE(reg.deformation_field, bwd_reg.deformation_field)
            metrics = {"ICE": ICE}
            
        else:
            metrics = {}
        
        # Save evaluation metrics
        SAVE_PATH = f"{args.result_path}/{i}"
        reg.evaluation(
            SAVE_PATH,
            metrics
            )
    
    # Summarize all metrics across dataset
    summarize_metrics(args, rm_results=False)


if __name__ == "__main__":
    main()
