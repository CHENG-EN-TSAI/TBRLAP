# TBR-LAP / TBRJ-LAP  
**Trigonometric-Basis Regularized Local All-Pass Registration for X-ray Images**

This repository provides an implementation of **LAP-based deformable 2D–2D medical image registration**, including **Original LAP**, **TBR-LAP**, and optional **TBRJ-LAP** with Jacobian/area constraints. The code targets **same-modality X-ray registration** and supports forward–backward registration with **inverse consistency error (ICE)** evaluation.

## Features
- Local All-Pass (LAP) deformation estimation
- Global deformation reconstruction using trigonometric (Fourier sine/cosine) basis  
- First- and second-order smoothness regularization  
- Optional Jacobian determinant constraints for folding suppression and area control  
- Forward + backward registration with ICE  
- Quantitative evaluation: SSIM, NMI, Dice, TV, Jacobian determinant, ICE  

## Repository Structure
.
├─ main.py
├─ lap_proposed.py
├─ lap_original.py
├─ lap_test.py
├─ functions.py
├─ data/
│  └─ NIH/
│     └─ NIH_regist_pair.json
└─ LAP_results/

## Installation
Python ≥ 3.9 recommended.

Dependencies:
numpy, pillow, tqdm, torch, scikit-image

Install:
pip install numpy pillow tqdm torch scikit-image

Note: ants_affine_registration() requires ANTs / ANTsPy. Please install ANTs accordingly.

## Data Preparation
--data_paths_json must point to a JSON file containing a list of dictionaries with keys:
fixed, moving, fixed_seg, moving_seg, fixed_size, moving_size,
fixed_piel_spacing, moving_piel_spacing.

Images are resized to --img_size (default 256×256), normalized to [0,1].
Masks are resized with nearest-neighbor and binarized.
Pixel spacing and size are used to compute spacing ratio.

## Usage
Run TBR-LAP:
python main.py --data_paths_json data/NIH/NIH_regist_pair.json --LAP_type proposed --device auto --result_path LAP_results --img_size 256 256 --r_list 8 8 8 4 4 2 2 1 1 --number_of_F_basis 5 --beta 5 150000 --gamma 0

Run original LAP:
python main.py --data_paths_json data/NIH/NIH_regist_pair.json --LAP_type original --device auto --result_path LAP_results

## Arguments
--data_paths_json: dataset JSON  
--img_size: image resolution  
--LAP_type: proposed | original | test  
--device: cpu | cuda | auto  
--result_path: output directory  
--sigma: Gaussian parameterization  
--r_list: multi-scale filter radii  
--number_of_F_basis: truncation parameter n  
--beta: smoothness regularization weights  
--gamma: Jacobian / area constraint weight  

## Workflow
1. Load fixed/moving images and segmentation masks  
2. Resize and normalize  
3. Initialize deformation via ANTs affine registration  
4. Forward registration  
5. Backward registration  
6. Compute ICE  
7. Save per-case metrics and visualizations  
8. Aggregate statistics across cases  

Note: The script currently runs only one pair (for i in [0]). Change to range(len(data_paths)) for full evaluation.

## Output
Results are saved to {result_path}/{i}/ including metrics.json and visualizations.
After completion, mean ± std of SSIM, NMI, Dice, ICE, TV_mean, TV_p95, DJ_fold%, DJ_p95, DJ_p5, and MSE[D_J-c] are printed.

## Evaluation Metrics
SSIM (structural similarity), NMI, Dice, TV (smoothness), Jacobian determinant (folding and area change), ICE.

## Citation
@mastersthesis{Tsai2026TBRLAP,
  title   = {A Trigonometric-Basis Regularized Local All-Pass Method with Component-wise Jacobian Constraints for X-ray Image Alignment},
  author  = {Cheng-En Tsai},
  school  = {National Taiwan University},
  year    = {2026}
}

## License
Specify license (e.g., MIT).

## Acknowledgements
Local All-Pass framework, ANTs / SimpleITK toolkits, NIH Chest X-ray dataset
