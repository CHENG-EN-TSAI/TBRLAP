import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from skimage.restoration import inpaint
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import normalized_mutual_information as nmi
from skimage.morphology import skeletonize
from PIL import Image as PImage
import json

# Custom modules
from functions import (
    apply_field, do_conv, pullback_update_field,
    intensity_transform,
    cuda_mean_filter, detect_body
)


__all__ = ["LAP"]

## an object for LAP registration
class LAP():

    """
    Attribution:
        im_shape:
        --tuple--
        shape of the imput image

        im_f:
        --np.array--
        the fixed img

        im_m:
        --np.array--
        the moving img

        im_w:
        --np.array-- 
        the transformed image (warped image + intensity transform)

        domain:
        --np.array--
        we only conside fixed image in this domain

        n:
        --int--
        number of the basis for filter

        R:
        --int--
        size of the filter

        filters:
        --list--
        basis of the filters

        deformation_field:
        --np.array--
        deformation field

        alpha:
        --function--
        intensity transformation

        mask:
        --np.array--
        the mask marking out the bad alignment region

        cont:
        --Boolean--
        a value determining whether continuing the process or not

        metrics: 
        --dict--
        store some metrics  of the result
    """


    def __init__(self,
                fixed_img: np.array,
                moving_img: np.array,
                args,
                initial_field=None,
                conserve_region=None,
                spacing_ratio=None
                ):

        """
        fixed_img and moving_img must satisfy 
        (i) same shape
        (ii) only two dimension
        e.g. (200, 300)
        """

        assert fixed_img.shape == moving_img.shape, "two images must have same shape"
        assert fixed_img.ndim == 2 and moving_img.ndim == 2, "images' dimension must be 2"
        
        self.im_shape = fixed_img.shape
        
        if initial_field is not None:
            assert initial_field.shape == (2, self.im_shape[0], self.im_shape[1]), "Initial field must have shape (2, H, W)"
            self.deformation_field = initial_field
            self.im_w = apply_field(moving_img, initial_field, order=3)
        else:
            self.deformation_field = np.zeros([2, self.im_shape[0], self.im_shape[1]])
            self.im_w = moving_img
            
        device = args.device
        global RUNNING_DEVICE
        RUNNING_DEVICE = device

        self.args = args
        self.spacing_ratio = spacing_ratio
        self.conserve_region = conserve_region
        self.im_shape = fixed_img.shape
        
        self.im_f = fixed_img
        self.im_m = moving_img
        self.body_f = detect_body(fixed_img)
        self.body_m = detect_body(moving_img)
        
        self.n = 3
        self.domain = self.body_m
        self.R = None
        self.sigma = None
        self.filters = None
        self.alpha = lambda image: image
        self.cont = True
        self.itr_times = 0
        
    # -------------------- Main iteration functions -------------------- #
    def run_iterations(
        self
        ):
        """Run LAP iterations over a list of filter sizes (multi-scale registration)."""
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
        sigma = (R+2)/4

        # define the basis of the filter
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

        # Apply the deformation field as the initial field
        im_f = self.im_f*domain
        im_m = apply_field(self.im_m, self.deformation_field)*domain

        # intensity transformation
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

        p_app = np.repeat(p[np.newaxis, 0, :, :], A.shape[0], axis=0)
        for i in range(self.n-1):
            c_tiles = c[:, i, np.newaxis, np.newaxis]  # shape = N X 1 X 1  
            p_tiles = np.repeat(p[np.newaxis, i+1, :, :], A.shape[0], axis=0)  # shape = N X (n-1) X (n-1)
            p_app += c_tiles*p_tiles
            
        Y, X = np.meshgrid(np.arange(-R, R+1), np.arange(-R, R+1))

        u0 = do_conv(p_app, X, R, device=RUNNING_DEVICE)
        u1 = do_conv(p_app, Y, R, device=RUNNING_DEVICE)

        filter_sum = np.sum(p_app, axis=(1, 2)) + 1e-8

        u0 = 2*u0/filter_sum
        u1 = 2*u1/filter_sum

        # Store the update field
        update_field_w = R*np.ones([2, self.im_shape[0], self.im_shape[1]])

        d = np.where(p[0]>0, 1, 0)
        D = (do_conv(domain, d, R, device=RUNNING_DEVICE) == np.sum(d))

        fill_ratio = 1 - (np.sum(D))/(self.im_shape[0]*self.im_shape[1])

        update_field_w[0, R:-R, R:-R] = np.where(D, u0.reshape(im_m.shape[0]-2*R, im_m.shape[1]-2*R), R)
        update_field_w[1, R:-R, R:-R] = np.where(D, u1.reshape(im_m.shape[0]-2*R, im_m.shape[1]-2*R), R)
        
        # Pullback the update field
        update_field = pullback_update_field(self.deformation_field, update_field_w)
        
        # Post-processing
        displacement = np.sqrt((update_field[0])**2 + (update_field[1])**2)

        mask = np.where(displacement >= self.R, False, True)
        
        # Stop iteration if too much of the image is not aligned or fill ratio too low
        if (self.itr_times > 0 and mask.mean() + fill_ratio < 0.5) or mask.mean() == 0:
            self.cont = False

        else:
            update_field[0] = inpaint.inpaint_biharmonic(update_field[0], 1-mask)
            update_field[1] = inpaint.inpaint_biharmonic(update_field[1], 1-mask)
            update_field = cuda_mean_filter(update_field, r=2*self.R, device=RUNNING_DEVICE, valid=False)
            
            self.deformation_field += update_field
            self.deformation_field = cuda_mean_filter(self.deformation_field, r=4*self.R, device=RUNNING_DEVICE, valid=False)

            self.domain = apply_field(self.body_m, self.deformation_field, order=0)

        self.itr_times += 1

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
        
        # Save GIF animation
        frames = []
        frames.append(PImage.fromarray(self.im_f*255))
        frames.append(PImage.fromarray(self.im_w*255))
        frames[0].save(os.path.join(path, 'anime.gif'), save_all=True, append_images=frames[1:], duration=500, loop=0)

        ux, uy = np.gradient(self.deformation_field[0], axis=0), np.gradient(self.deformation_field[0], axis=1)
        vx, vy = np.gradient(self.deformation_field[1], axis=0), np.gradient(self.deformation_field[1], axis=1)

        # Compute deformation gradients and TV
        TV = np.sqrt(ux**2 + \
                    uy**2 + \
                    vx**2 + \
                    vy**2)
        
        D_J = ((1+ ux)*(1+vy) - uy*vx)

        # Save images, TV, and absolute error
        images_dict = {"Fixed Image": self.im_f,
                    "Moving Image": self.im_m,
                    "Warped Image": self.im_w
                    }

        for name, image in images_dict.items():

            if name == "TV":
                plt.imshow(image, cmap="gray", vmin=0)
            else:
                plt.imshow(image, cmap="gray", vmin=0, vmax=1)

            plt.axis("off")

            if name == "TV":
                plt.colorbar()

            plt.savefig(os.path.join(path, name), bbox_inches="tight", pad_inches=0)
            plt.close()
        
        step = 17
        
        u, v = self.deformation_field[0], self.deformation_field[1]
        H, W = u.shape

        # coordinate grid
        X, Y = np.mgrid[:self.im_shape[0]:step, :self.im_shape[1]:step]

        # FIXED: Map u to horizontal and v to vertical
        plt.imshow(TV, vmin=0)
        plt.axis("off")
        plt.colorbar()
        
        plt.clim(vmin=0, vmax=0.6)
        
        # Your custom limits to prevent arrow clipping at boundaries
        plt.xlim([-24, W + 24])
        plt.ylim([H + 24, -25]) # Keeps origin at top-left
        
        # Use angles='xy' so arrows align with the pixel grid
        plt.quiver(Y, X, v[::step, ::step], -u[::step, ::step], color="r", headwidth=4, scale_units='xy')
        
        plt.savefig(os.path.join(path, "Deformation Field"), bbox_inches="tight", pad_inches=0)
        plt.close()
        
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

        # Plot
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
        
        # Compute SSIM, NMI, Jacobian metrics

        if self.spacing_ratio is not None:
            MSE_DJ_c = float(((D_J - self.spacing_ratio)[self.conserve_region==1]**2).mean())
        else:
            MSE_DJ_c = None
            
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
