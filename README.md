# TBR-LAP / TBRJ-LAP  
**Trigonometric-Basis Regularized Local All-Pass Registration for 2D Same-modal Images**

This repository provides an implementation of **LAP-based deformable 2D–2D medical image registration**, including **Original LAP**, **TBR-LAP**, and optional **TBRJ-LAP** with Jacobian/area constraints. The code targets **same-modality X-ray registration** and supports forward–backward registration with **inverse consistency error (ICE)** evaluation.

## Features
- Local All-Pass (LAP) deformation estimation
- Global deformation reconstruction using trigonometric (Fourier sine/cosine) basis  
- First- and second-order smoothness regularization  
- Optional Jacobian determinant constraints for folding suppression and area control  
- Forward + backward registration with ICE  
- Quantitative evaluation: SSIM, NMI, Dice, TV, Jacobian determinant, ICE  

## Repository Structure
```text
.
├─ go.py
├─ lap_proposed.py
├─ lap_original.py
├─ lap_test.py
├─ functions.py
└─ test_data
```

## Installation
1. Clone the repo
   ```text
   git clone https://github.com/CHENG-EN-TSAI/TBRLAP.git
   ```
   
2. Create new conda environment
   ```text
   conda create -n oxr python==3.9
   conda activate lap
   ```
   
3. Install required packages
   ```text
   pip install numpy pillow tqdm torch scikit-image
   ```
4. Install ANTs / ANTsPy follow https://

## Data Preparation

`--data_paths_json` must point to a JSON file containing a **list of dictionaries**.
Each dictionary specifies one registration pair.

**Required fields**
- `fixed`: path to the fixed image
- `moving`: path to the moving image

**Optional fields**
- `moving_seg`: path to the moving image segmentation (binary mask)
- `fixed_piel_spacing`: pixel spacing of the fixed image `[sx, sy]`
- `moving_piel_spacing`: pixel spacing of the moving image `[sx, sy]`

### Example JSON format

```json
[
  {
    "fixed": "FIXED_IMAGE_PATH",
    "fixed_piel_spacing": [sx, sy],
    "moving": "MOVING_IMAGE_PATH",
    "moving_seg": "MOVING_SEG_PATH",
    "moving_piel_spacing": [sx, sy]
  }
]
```

## RUN A Test
Run TBR-LAP:
```text
python main.py --LAP_type proposed
```

Run original LAP:
```text
python main.py --LAP_type original
```

## Arguments
--data_paths_json: dataset JSON  
--img_size: image resolution  
--LAP_type: proposed | original 
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

## Contact
Any problem pliease contact 
