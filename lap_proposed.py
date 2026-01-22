import os
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy import linalg
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import normalized_mutual_information as nmi
from skimage.morphology import skeletonize
from skimage.segmentation import find_boundaries
from PIL import Image as PImage
import warnings

# Custom modules
from functions import (
    apply_field, do_conv, pullback_update_field,
    intensity_transform,
    detect_body
)

__all__ = ["LAP"]

## an object for LAP registration
class LAP():
    """
    Local All-Pass (LAP) image registration class.

    Attributes:
        im_shape: tuple, shape of input images
        fixed_image: np.array, fixed image
        moving_image: np.array, moving image
        warped_image: np.array, transformed moving image after registration
        domain: np.array, region to consider for registration
        n: int, number of sine-cosine basis filters
        R: int, current filter radius
        filters: list, filter basis [forward, backward]
        deformation_field: np.array, 2xHxW deformation vectors
        alpha: callable, intensity transformation function
        mask: np.array, region mask for bad alignment
        cont: bool, continue iteration flag
        metrics: dict, stores evaluation metrics
    """

    def __init__(
        self,
        fixed_img: np.array,
        moving_img: np.array,
        args,
        initial_field=None,
        conserve_region=None,
        spacing_ratio=None
        ):

        """
        Initialize LAP registration object.

        Args:
            fixed_image: HxW fixed image
            moving_image: HxW moving image
            device: "cpu", "cuda", or "auto"
            body_mask_m: optional mask of moving image body
            conserve_region: optional mask for volume preservation
            spacing_ratio: optional spacing ratio for volume conservation
        """
        assert fixed_img.shape == moving_img.shape, "Images must have same shape"
        assert fixed_img.ndim == 2 and moving_img.ndim == 2, "Images must be 2D"
        assert args.beta[0] > 0, "beta_1 must be positive"
        assert args.beta[1] >= 0, "beta_2 must be non-negative"
        assert args.gamma >= 0, "gamma must be non-negative"
        
        self.im_shape = fixed_img.shape
        
        if initial_field is not None:
            assert initial_field.shape == (2, self.im_shape[0], self.im_shape[1]), "Initial field must have shape (2, H, W)"
            self.deformation_field = initial_field
            self.im_w = apply_field(moving_img, initial_field, order=3)
        else:
            self.deformation_field = np.zeros([2, self.im_shape[0], self.im_shape[1]])
            self.im_w = moving_img

        if conserve_region is not None:
            assert fixed_img.shape == conserve_region.shape, "Conserved region must have same shape as images"
        
        if spacing_ratio is None and args.gamma > 0:
            warnings.warn("Warning: spacing_ratio is not provided, volume conservation will be disabled even if gamma > 0")

        device = args.device
        global RUNNING_DEVICE
        RUNNING_DEVICE = device

        self.spacing_ratio = spacing_ratio
        self.conserve_region = conserve_region
        self.args = args
        
        self.im_f = fixed_img
        self.im_m = moving_img
        self.body_f = detect_body(fixed_img)
        self.body_m = detect_body(moving_img)
        
        self.itr_times = 0
        self.n = 3
        self.domain = self.body_m
        self.R = None
        self.sigma = None
        self.filters = None
        self.alpha = lambda image: image
        self.cont = True
        
        # For post-processing
        self.SC = None
        self.SCx = None
        self.SCy = None
        self.SCxx = None
        self.SCxy = None
        self.SCyy = None
        self.Mx = None
        self.My = None
        self.Mxx = None
        self.Mxy = None
        self.Myy = None
        
    # -------------------- Main iteration functions -------------------- #
    def run_iterations(self):
        """Run LAP iterations over a list of filter sizes."""
        for R in self.args.r_list:
            if self.cont is True:
                self.one_iteration(R)
            else:
                break
        
        self.im_w = apply_field(self.im_m, self.deformation_field, order=3)

    def one_iteration(self,
                    R: int
                    ):
        """
        Perform a single LAP iteration for a given filter size.
        Updates deformation_field and intensity transform.
        """        
        self.R = R
        
        if self.args.sigma == "original":
            sigma = (R+2)/4
        elif self.args.sigma == "proposed":
            sigma = np.sqrt(-0.5/np.log(0.01/2/np.pi/R))*R

        # Create filter basis if sigma changed
        if self.sigma == sigma:
            p, p_n = self.filters

        else:
            self.sigma = sigma
            p = np.zeros([self.n, 2*R+1, 2*R+1])
            p_n = np.zeros([self.n, 2*R+1, 2*R+1])
            for i in range(-R, R+1, 1):
                for j in range(-R, R+1, 1):
                    #if i**2+j**2 <= R**2+1: 
                    a = np.exp(-(i**2+j**2)/2/sigma**2)
                    p[0,i+R,j+R] = a
                    p[1,i+R,j+R] = i*a
                    p[2,i+R,j+R] = j*a
                    p_n[1,i+R,j+R] = -i*a
                    p_n[2,i+R,j+R] = -j*a

            p_n[0] = p[0]
            self.filters = [p, p_n]

        domain = (self.domain.astype(bool) & self.body_f.astype(bool))

        # Apply current deformation
        im_f = self.im_f*domain
        im_m = apply_field(self.im_m, self.deformation_field)*domain

        # Update intensity transformation
        im_m, _ = intensity_transform(im_f, im_m)

        # Obtain and solve the linear systems
        B = np.zeros([self.n, self.im_shape[0], self.im_shape[1]])

        for i in range(self.n):
            B[i] = do_conv(im_f, p[i], R, device=RUNNING_DEVICE, valid=False) - do_conv(im_m, p_n[i], R, device=RUNNING_DEVICE, valid=False)

        A = np.zeros([(self.im_shape[0]-2*R)*(self.im_shape[1]-2*R), self.n-1, self.n-1])
        b = np.zeros([A.shape[0], self.n-1])

        ones_kernel = np.ones([2*R + 1, 2*R + 1])

        for i in range(self.n-1):
            b[:, i] = do_conv(B[0]*B[i+1], ones_kernel, R, device=RUNNING_DEVICE).reshape(-1)
            A[:, i, i] = do_conv(B[i+1]*B[i+1], ones_kernel, R, device=RUNNING_DEVICE).reshape(-1)
            for j in range(i+1, self.n-1):
                A[:, i, j] = do_conv(B[i+1]*B[j+1], ones_kernel, R, device=RUNNING_DEVICE).reshape(-1)
                A[:, j, i] = A[:, i, j]

        c = torch.linalg.lstsq(torch.tensor(A), torch.tensor(b)).solution.numpy()
        
        # Compute the forward filter
        p_app = np.repeat(p[np.newaxis, 0, :, :], A.shape[0], axis=0)
        for i in range(self.n-1):
            c_tiles = c[:, i, np.newaxis, np.newaxis]  # shape = N X 1 X 1  
            p_tiles = np.repeat(p[np.newaxis, i+1, :, :], A.shape[0], axis=0)  # shape = N X (n-1) X (n-1)
            p_app += c_tiles*p_tiles

        # Compute the update field
        X, Y = np.meshgrid(np.arange(-R, R+1), np.arange(-R, R+1), indexing="ij")
        

        u0 = do_conv(p_app, X, R, device="cpu")
        u1 = do_conv(p_app, Y, R, device="cpu")

        filter_sum = np.sum(p_app, axis=(1, 2)) + 1e-8

        u0 = 2*u0/filter_sum
        u1 = 2*u1/filter_sum

        # Store the update field
        update_field_w = R*np.ones([2, self.im_shape[0], self.im_shape[1]])

        d = np.where(p[0]>0, 1, 0)
        D = (do_conv(domain, d, R, device="cpu") == np.sum(d))

        fill_ratio = 1 -  (np.sum(D))/(self.im_shape[0]*self.im_shape[1])
            

        update_field_w[0, R:-R, R:-R] = np.where(D, u0.reshape(im_m.shape[0]-2*R, im_m.shape[1]-2*R), R)
        update_field_w[1, R:-R, R:-R] = np.where(D, u1.reshape(im_m.shape[0]-2*R, im_m.shape[1]-2*R), R)
        
        # Pullback the update field
        update_field = pullback_update_field(self.deformation_field, update_field_w)

        # Generate mask of valid displacement
        displacement = np.sqrt((update_field[0])**2 + (update_field[1])**2)

        mask = np.where(displacement >= self.R, False, True)
        
        # Stop iteration if too much of the image is not aligned or fill ratio too low
        if (self.itr_times > 0 and mask.mean() + fill_ratio < 0.5) or mask.mean() == 0:
            self.cont = False
        else:
            self.post_processing(update_field, mask)

            self.domain = apply_field(self.body_m, self.deformation_field, order=0)

        self.itr_times += 1

    def create_matrixes(self, update_field, axis, mask):
        
        N = self.args.number_of_F_basis
        beta = self.args.beta
        
        if self.SC is None or self.SC.shape[0] != 4*N**2:
            n, m = self.im_shape
            X, Y = np.meshgrid(np.arange(n), np.arange(m), indexing='ij')
            
            cx, cy = np.pi/n, np.pi/m
            
            X = cx*X
            Y = cy*Y

            SC, SCx, SCy = np.zeros([4*N**2, n, m]), np.zeros([4*N**2, n, m]), np.zeros([4*N**2, n, m])
            SCxx, SCxy, SCyy = np.zeros([4*N**2, n, m]), np.zeros([4*N**2, n, m]), np.zeros([4*N**2, n, m])

            n -= 1
            m -= 1

            for i in range(N):
                for j in range(N):
                    k = i*N + j
                    SC[k] = np.sin(X*(i+1))*np.sin(Y*(j+1))
                    SC[k + 1*N**2] = np.sin(X*(i+1))*np.cos(Y*j)
                    SC[k + 2*N**2] = np.cos(X*i)*np.sin(Y*(j+1))
                    SC[k + 3*N**2] = np.cos(X*i)*np.cos(Y*j)

                    SCx[k] = cx*(i+1)*np.cos(X*(i+1))*np.sin(Y*(j+1))
                    SCx[k + 1*N**2] = cx*(i+1)*np.cos(X*(i+1))*np.cos(Y*j)
                    SCx[k + 2*N**2] = -i*cx*np.sin(X*i)*np.sin(Y*(j+1))
                    SCx[k + 3*N**2] = -i*cx*np.sin(X*i)*np.cos(Y*j)

                    SCy[k] = cy*(j+1)*np.sin(X*(i+1))*np.cos(Y*(j+1))
                    SCy[k + 1*N**2] = -j*cy*np.sin(X*(i+1))*np.sin(Y*j)
                    SCy[k + 2*N**2] = cy*(j+1)*np.cos(X*i)*np.cos(Y*(j+1))
                    SCy[k + 3*N**2] = -j*cy*np.cos(X*i)*np.sin(Y*j)
                    
                    SCxx[k] = -((i+1)*cx)**2*np.sin(X*(i+1))*np.sin(Y*(j+1))
                    SCxx[k + 1*N**2] = -((i+1)*cx)**2*np.sin(X*(i+1))*np.cos(Y*j)
                    SCxx[k + 2*N**2] = -(i*cx)**2*np.cos(X*i)*np.sin(Y*(j+1))
                    SCxx[k + 3*N**2] = -(i*cx)**2*np.cos(X*i)*np.cos(Y*j)
                    
                    SCxy[k] = (j+1)*cy*(i+1)*cx*np.cos(X*(i+1))*np.cos(Y*(j+1))
                    SCxy[k + 1*N**2] = -j*cy*(i+1)*cx*np.cos(X*(i+1))*np.sin(Y*j)
                    SCxy[k + 2*N**2] = -(j+1)*cy*i*cx*np.sin(X*i)*np.cos(Y*(j+1))
                    SCxy[k + 3*N**2] = j*cy*i*cx*np.sin(X*i)*np.sin(Y*j)

                    SCyy[k] = -((j+1)*cy)**2*np.sin(X*(i+1))*np.sin(Y*(j+1))
                    SCyy[k + 1*N**2] = -(j*cy)**2*np.sin(X*(i+1))*np.cos(Y*j)
                    SCyy[k + 2*N**2] = -((j+1)*cy)**2*np.cos(X*i)*np.sin(Y*(j+1))
                    SCyy[k + 3*N**2] = -(j*cy)**2*np.cos(X*i)*np.cos(Y*j)

            self.SC = SC
            self.SCx = SCx
            self.SCy = SCy
            self.SCxx = SCxx
            self.SCxy = SCxy
            self.SCyy = SCyy
            
            self.Mx = SCx.reshape(4*N**2, -1)@SCx.reshape(4*N**2, -1).T
            self.My = SCy.reshape(4*N**2, -1)@SCy.reshape(4*N**2, -1).T
            self.Mxx = SCxx.reshape(4*N**2, -1)@SCxx.reshape(4*N**2, -1).T
            self.Mxy = SCxy.reshape(4*N**2, -1)@SCxy.reshape(4*N**2, -1).T
            self.Myy = SCyy.reshape(4*N**2, -1)@SCyy.reshape(4*N**2, -1).T
                    
        SC2 = self.SC[:, mask]
        A = (
            SC2@SC2.T/mask.mean()
            + beta[0]*(self.Mx + self.My)
            + beta[1]*(self.Mxx + self.Myy + 2*self.Mxy)
        )
        
        ux = np.gradient(self.deformation_field[axis], axis=0)
        uy = np.gradient(self.deformation_field[axis], axis=1)
        
        dx = np.sum(self.SCx*ux[np.newaxis, :, :], axis=(1, 2))
        dy = np.sum(self.SCy*uy[np.newaxis, :, :], axis=(1, 2))
        
        dxx = np.sum(self.SCxx*np.gradient(ux, axis=0)[np.newaxis, :, :], axis=(1, 2))
        dxy = np.sum(self.SCxy*np.gradient(ux, axis=1)[np.newaxis, :, :], axis=(1, 2))
        dyy = np.sum(self.SCyy*np.gradient(uy, axis=1)[np.newaxis, :, :], axis=(1, 2))

        b = (
            np.sum(self.SC*(update_field*mask)[np.newaxis, :, :], axis=(1, 2))/mask.mean()
            - beta[0]*(dx + dy) 
            - beta[1]*(dxx + dyy + 2*dxy)
        )

        return A, b

    
    # -------------------- Post-processing -------------------- #
    def post_processing(self, update_field, mask):
        """Update deformation field with regularization and optional volume conservation"""
        i = (self.itr_times+1)%2

        A, b = self.create_matrixes(update_field[i], i, mask)
        par = linalg.solve(A, b)
        self.deformation_field[i] += np.sum(par[:, np.newaxis, np.newaxis]*self.SC, axis=0)

        i = self.itr_times%2

        A, b = self.create_matrixes(update_field[i], i, mask)

        if self.spacing_ratio is None or self.args.gamma==0:
            par = linalg.solve(A, b)
            self.deformation_field[i] += np.sum(par[:, np.newaxis, np.newaxis]*self.SC, axis=0)

        else:
            ux = np.gradient(self.deformation_field[0], axis=0)[np.newaxis, :, :] + 1
            uy = np.gradient(self.deformation_field[0], axis=1)[np.newaxis, :, :]
            vx = np.gradient(self.deformation_field[1], axis=0)[np.newaxis, :, :]
            vy = np.gradient(self.deformation_field[1], axis=1)[np.newaxis, :, :] + 1

            if self.conserve_region is not None:
                SCx = np.where(self.conserve_region==1, self.SCx, 0)
                SCy = np.where(self.conserve_region==1, self.SCy, 0)
                constant = 1
                
            else:
                SCx = self.SCx
                SCy = self.SCy
                constant = 1/self.conserve_region.mean()
                
            N = self.args.number_of_F_basis
            gamma = self.args.gamma

            if i == 0:
                b += constant*gamma*np.sum((ux*vy-uy*vx-self.spacing_ratio)*(vx*SCy - vy*SCx), axis=(1,2))
                A += constant*gamma*((SCx*vy**2).reshape(4*N**2, -1)@SCx.reshape(4*N**2, -1).T \
                            + (SCy*vx**2).reshape(4*N**2, -1)@SCy.reshape(4*N**2, -1).T  \
                            - (SCx*vx*vy).reshape(4*N**2, -1)@SCy.reshape(4*N**2, -1).T \
                            - (SCy*vx*vy).reshape(4*N**2, -1)@SCx.reshape(4*N**2, -1).T)

                par = linalg.solve(A, b)
                self.deformation_field[i] += np.sum(par[:, np.newaxis, np.newaxis]*self.SC, axis=0)

            else:
                b += constant*gamma*np.sum((ux*vy-uy*vx-self.spacing_ratio)*(uy*SCx - ux*SCy), axis=(1,2))
                A += constant*gamma*((SCy*ux**2).reshape(4*N**2, -1)@SCy.reshape(4*N**2, -1).T \
                            + (SCx*uy**2).reshape(4*N**2, -1)@SCx.reshape(4*N**2, -1).T  \
                            - (SCx*uy*ux).reshape(4*N**2, -1)@SCy.reshape(4*N**2, -1).T \
                            - (SCy*uy*ux).reshape(4*N**2, -1)@SCx.reshape(4*N**2, -1).T)

                par = linalg.solve(A, b)
                self.deformation_field[i] += np.sum(par[:, np.newaxis, np.newaxis]*self.SC, axis=0)

    # -------------------- Evaluation -------------------- #
    def evaluation(self,
                path,
                metrics = {}
                ):
        """
        Evaluate registration, save images, deformation field, and metrics.

        Args:
            path: folder to save outputs
            metrics: dict with existing metrics
        """
            
        os.makedirs(path, exist_ok=True)
        
        ######################################
        #### ---- Save GIF animation ---- ####
        ######################################
        frames = []
        frames.append(PImage.fromarray(self.im_f*255))
        frames.append(PImage.fromarray(self.im_w*255))
        frames[0].save(os.path.join(path, 'anime.gif'), save_all=True, append_images=frames[1:], duration=500, loop=0)


        H, W = self.im_shape

        ###############################
        #### ---- Save images ---- ####
        ###############################
        images_dict = {"Fixed Image": self.im_f,
                    "Moving Image": self.im_m,
                    "Warped Image": self.im_w
                    }
        if self.conserve_region is not None:
            boundary = find_boundaries(self.conserve_region, mode='inner')

        for name, image in images_dict.items():
            if name!="Fixed Image" and self.conserve_region is not None:
                if name == "Moving Image":
                    plt.imshow(image, cmap="gray", vmin=0, vmax=1)
                    plt.contour(
                        boundary,
                        levels=[0.8],
                        colors=["r"],
                        linewidths=0.5,
                        origin="upper",
                        extent=(0, W, H, 0),
                    )
                    plt.axis("off")
                    plt.savefig(os.path.join(path, name), bbox_inches="tight", pad_inches=0)
                    plt.close()
                    
                elif name == "Warped Image":
                    plt.imshow(image, cmap="gray", vmin=0, vmax=1)
                    plt.contour(
                        apply_field(boundary, self.deformation_field, order=0),
                        levels=[0.8],
                        colors=["r"],
                        linewidths=0.5,
                        origin="upper",
                        extent=(0, W, H, 0),
                    )
                    plt.axis("off")
                    plt.savefig(os.path.join(path, name), bbox_inches="tight", pad_inches=0)
                    plt.close()
                    
            else:
                plt.imshow(image, cmap="gray", vmin=0, vmax=1)
                plt.axis("off")
                plt.savefig(os.path.join(path, name), bbox_inches="tight", pad_inches=0)
                plt.close()
        
        ##########################################
        #### ---- Save deformation field ---- ####
        ##########################################
        ux, uy = np.gradient(self.deformation_field[0], axis=0), np.gradient(self.deformation_field[0], axis=1)
        vx, vy = np.gradient(self.deformation_field[1], axis=0), np.gradient(self.deformation_field[1], axis=1)
        
        TV = np.sqrt(ux**2 + \
                    uy**2 + \
                    vx**2 + \
                    vy**2)
        
        D_J = ((1+ ux)*(1+vy) - uy*vx)

        step = 17
        
        u, v = self.deformation_field[0], self.deformation_field[1]

        # coordinate grid
        X, Y = np.mgrid[:self.im_shape[0]:step, :self.im_shape[1]:step]
        
        for name in ["TV Map", "Jacobian Determinant"]:
            if name == "Jacobian Determinant":
                plt.imshow(D_J, vmin=0)
            elif name == "TV Map":
                plt.imshow(TV, vmin=0)

            plt.axis("off")
            plt.colorbar()
            
            if name == "TV Map":
                plt.clim(vmin=0, vmax=0.5)
                
            elif name == "Jacobian Determinant":
                plt.clim(vmin=0.8, vmax=1.2)
            
            # Your custom limits to prevent arrow clipping at boundaries
            plt.xlim([-24, W + 24])
            plt.ylim([H + 24, -25]) # Keeps origin at top-left
            
            # Use angles='xy' so arrows align with the pixel grid
            plt.quiver(Y, X, v[::step, ::step], -u[::step, ::step], color="r", headwidth=4, scale_units='xy')
            
            plt.savefig(os.path.join(path, f"Deformation Field on {name}"), bbox_inches="tight", pad_inches=0)
            plt.close()
        
        ###############################################
        #### ---- Save warped image with grid ---- ####
        ###############################################
        spacing = 32
        
        grid = np.zeros((H, W), dtype=np.float32)
        grid[::spacing, :] = 1.0
        grid[:, ::spacing] = 1.0
        grid[1::spacing, :] = 1.0
        grid[:, 1::spacing] = 1.0
        grid[2::spacing, :] = 1.0
        grid[:, 2::spacing] = 1.0
        grid[-1, :] = 1.0
        grid[:, -1] = 1.0
        
        warped_grid = skeletonize(
            apply_field(grid, np.stack([u, v], axis=0), order=1)
        )

        fig = plt.figure(figsize=(W / 100, H / 100), dpi=200)  # scale with image size
        ax = plt.gca()

        ax.imshow(self.im_w, cmap="gray", alpha=1.0)
            
        ax.contour(
            warped_grid,
            levels=[0.8],
            colors=["r"],
            linewidths=0.5,
            origin="upper",
            extent=(0, W, H, 0),
        )
        
        ax.invert_yaxis()   # match image coordinates
        ax.set_xlim([0, W - 1])
        ax.set_ylim([H - 1, 0])
        ax.axis("off")

        fig.savefig(os.path.join(path, "Warped Grid"), bbox_inches="tight", pad_inches=0)
        plt.close()
        
        
        #################################################
        #### ---- save J_\phi - r with boundary ---- ####
        #################################################
        if self.spacing_ratio is not None and self.conserve_region is not None:
            DJ_c = D_J - self.spacing_ratio
            MSE_DJ_c = float((DJ_c[self.conserve_region==1]**2).mean())
            plt.imshow(DJ_c, vmin=-0.15, vmax=0.15, cmap="seismic")
            plt.axis("off")
            plt.colorbar()
            plt.contour(
                boundary,
                levels=[0.8],
                colors=["k"],
                linewidths=0.5,
                origin="upper",
                extent=(0, W, H, 0),
            )
            plt.savefig(os.path.join(path, "Jacobian Error"), bbox_inches="tight", pad_inches=0)
            plt.close()
            
        else:
            MSE_DJ_c = None
            
        ############################################
        #### ---- Compute and save metrics ---- ####
        ############################################
        valid = np.where(apply_field(np.ones_like(self.im_m), self.deformation_field, order=0)==1, True, False)
        _, SSIM_img = ssim(self.im_f, self.im_w, data_range=1.0, full=True)
        SSIM_val = float(np.mean(SSIM_img[valid]))
        NMI_val  = float(nmi(self.im_f[valid], self.im_w[valid]))

        metrics["SSIM"] = float(SSIM_val)
        metrics["NMI"] = float(NMI_val)
        metrics["TV_mean"] = float(TV.mean())
        metrics["TV_p95"] = float(np.percentile(TV, 95))
        metrics["DJ_fold%"] = float((D_J <= 0).mean() * 100.0)
        metrics["DJ_p95"] = float(np.percentile(D_J, 95))
        metrics["DJ_p5"] = float(np.percentile(D_J, 5))
        metrics["MSE[D_J-c]"] = MSE_DJ_c

        with open(os.path.join(path, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
