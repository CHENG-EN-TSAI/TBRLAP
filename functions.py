import numpy as np
import torch.nn as nn
from scipy import ndimage as ndi
from scipy import linalg
from skimage.transform import warp, resize
import ants
import pandas as pd
import torch

__all__ = ["apply_field", "do_conv", "pullback_update_field", 
           "cuda_gaussian_filter", "cuda_mean_filter",
           "detect_body", "intensity_transform",
           "dice_score", "compute_ICE", "ants_affine_registration"
           ]

## apply a deformation field on an image
def apply_field(im: np.array, deformation_field: np.array, order=1) -> np.array:
    X, Y = np.meshgrid(np.arange(im.shape[0]), np.arange(im.shape[1]), indexing='ij')
    im_warp = warp(im, np.array([X + deformation_field[0], Y + deformation_field[1]]), order=order, preserve_range=True)
    return im_warp

## do convolution using pytorch and GPU if available
def do_conv(image: np.array, kernel: np.array, R, device="cpu", valid=True):

    dim = image.ndim

    filter = torch.tensor(kernel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    if dim == 2:
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    elif dim  == 3:
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(1)
    
    if valid is True:
        conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2*R+1, padding="valid", bias=False)
        conv.weight = nn.Parameter(filter)
    else:
        conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2*R+1, padding='same', bias=False, padding_mode="reflect")
        conv.weight = nn.Parameter(filter)

    conv.eval()
    with torch.no_grad():
        conv = conv.to(device)
        image = image.to(device)
        output = conv(image)

    if dim == 2:
        return output.cpu().numpy()[0][0]

    elif dim == 3:
        return output.cpu().numpy()[:,0,0,0]

# pull back warped update field to original grid
def pullback_update_field(deformation_field, update_field_w):
    _, H, W = deformation_field.shape
    X, Y = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    X_w = X + deformation_field[0]
    Y_w = Y + deformation_field[1]
    # sample each component
    out = np.zeros_like(deformation_field)
    for i in range(2):
        out[i] = ndi.map_coordinates(
            update_field_w[i],
            [X_w, Y_w],
            order=0, mode='nearest', prefilter=False
        )
    return out

def cuda_mean_filter(img: np.array, r: int, device="cuda", valid=True, Normalize=True):
    filter = np.ones([2*r+1, 2*r+1])

    if Normalize is True:
        filter = filter/filter.sum()

    filter = torch.tensor(filter, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    if img.ndim==2:
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    else:
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(1)

    if valid is True:
        conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2*r+1, padding="valid", bias=False)
    else:
        conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2*r+1, padding='same', bias=False, padding_mode="reflect")

    conv.weight = nn.Parameter(filter)

    conv.eval()
    with torch.no_grad():
        conv = conv.to(device)
        img = img.to(device)
        output = conv(img)

    return output.cpu().numpy().squeeze()

#roughly detect the image background
def detect_body(image):

    background = np.where((np.gradient(image, axis=0)==0) & (np.gradient(image, axis=1)==0), 1, 0)

    # Connected component labeling
    structure = ndi.generate_binary_structure(2, 1)
    labeled_image, num_features = ndi.label(background, structure=structure)

    if num_features == 0:
        return np.ones_like(image)

    # Compute component sizes
    component_sizes = np.bincount(labeled_image.ravel())

    # Remove the background found smaller thean the certain size
    min_size = 0.01*image.shape[0]*image.shape[1]
    for i, size in enumerate(component_sizes):
        if size < min_size:
            background = np.where(labeled_image==i, 0, background)

    #structure = ndi.generate_binary_structure(2, 2)
    #background = ndi.binary_dilation(background, structure=structure, iterations=1)

    return np.where(background==0, 1, 0)

def intensity_transform(im_f: np.array,
                        im_m: np.array
                        ):
        
    n, m = im_f.shape
    X, Y = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, m), indexing='ij')

    x = X.ravel()
    y = Y.ravel()
    f = im_f.ravel()
    g = im_m.ravel()

    B = np.stack([x * g, y * g, g], axis=1)

    # normal equations
    A = B.T @ B

    # check conditioning
    if np.linalg.cond(A) > 1e8:
        alpha = lambda image: image
        return im_m, alpha

    # solve for coefficients
    c = np.linalg.solve(A, B.T @ f)

    # define intensity transform
    alpha = lambda image: (c[0] * X + c[1] * Y + c[2]) * image
    
    return alpha(im_m), alpha

def dice_score(binA, binB, eps=1e-8):
    A = (binA > 0.5).astype(np.float32)
    B = (binB > 0.5).astype(np.float32)
    inter = float((A*B).sum())
    return (2*inter) / (float(A.sum() + B.sum()) + eps)

def compute_ICE(u_fwd_fix, u_bwd_mov, spacing_fix=(1.0,1.0), spacing_mov=(1.0,1.0)):
    """
    ICE = mean_{(r,c) in valid} || u_fwd_fix(r,c) + u_bwd_mov(r',c') ||_2
    with (r',c') = (r + u_fwd_fix_row, c + u_fwd_fix_col)

    u_fwd_fix: (2,Hf,Wf) forward field (fixed→moving) in *pixels* [dr, dc] on FIXED grid
    u_bwd_mov: (2,Hm,Wm) backward field (moving→fixed) in *pixels* [dr, dc] on MOVING grid
    spacing_*: (sr, sc) pixel spacings (row, col) to optionally compute ICE in physical units
    """
    uf = u_fwd_fix.astype(np.float32)
    ub = u_bwd_mov.astype(np.float32)

    Hf, Wf = uf.shape[1:]
    Hm, Wm = ub.shape[1:]

    R, C = np.indices((Hf, Wf), dtype=np.float32)  # rows, cols on FIXED grid
    Rq = R + uf[0]   # query rows on MOVING grid
    Cq = C + uf[1]   # query cols on MOVING grid

    valid = (Rq >= 0) & (Rq <= Hm - 1) & (Cq >= 0) & (Cq <= Wm - 1)
    if not np.any(valid):
        return float("nan")
    
    ub_at = np.zeros_like(uf)
    ub_at[0] = warp(ub[0], np.array([Rq, Cq]), order=1, preserve_range=True)
    ub_at[1] = warp(ub[1], np.array([Rq, Cq]), order=1, preserve_range=True)
    
    res = uf + ub_at  # ideal inverse-consistent ⇒ res ≈ 0

    # If you want ICE in physical units, scale components by spacings before the norm:
    sr_f, sc_f = spacing_fix
    # (Assuming uf/ub are in pixels of their own grids; if spacings differ, ideally
    # convert both to a common physical space before combining)
    res_phys_r = res[0] * sr_f
    res_phys_c = res[1] * sc_f

    ice_map = np.sqrt(res_phys_r**2 + res_phys_c**2)
    return float(ice_map[valid].mean())

def ants_affine_registration(fixed, moving, seed=False, stride=2):
    """
    Perform affine registration using ANTsPy.
    Args:
        fixed: fixed image (H,W) numpy array
        moving: moving image (H,W) numpy array
    Returns:
        deformation_field: (2, H, W) numpy array representing the deformation field
    """
    fixed_ants = ants.from_numpy(fixed)
    moving_ants = ants.from_numpy(moving)
    
    # Perform affine registration
    reg = ants.registration(fixed_ants, moving_ants, type_of_transform='Affine', random_seed=seed)
               
    transform_list = reg['fwdtransforms']
    
    # Compute deformation field by applying the transform to a grid of points
    H, W = fixed.shape
    Xs, Ys = np.meshgrid(np.arange(0, H, stride, dtype=np.float32),
                    np.arange(0, W, stride, dtype=np.float32),
                    indexing='ij')
    pts_df = pd.DataFrame({"x": Xs.ravel(), "y": Ys.ravel()})
    warped_df = ants.apply_transforms_to_points(dim=2, points=pts_df, transformlist=transform_list, whichtoinvert=[False])

    Xw = warped_df["x"].to_numpy().reshape(Ys.shape)
    Yw = warped_df["y"].to_numpy().reshape(Ys.shape)
    u = (Xw - Xs).astype(np.float32)
    v = (Yw - Ys).astype(np.float32)
    if stride != 1:
        u = resize(u, (H, W), order=1, preserve_range=True).astype(np.float32)
        v = resize(v, (H, W), order=1, preserve_range=True).astype(np.float32)

    return np.stack([u, v], axis=0)
