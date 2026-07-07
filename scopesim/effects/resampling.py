# -*- coding: utf-8 -*-
from typing import ClassVar

import numpy as np

from ..optics.fov import FieldOfView
from ..optics.image_plane import ImagePlane
from ..utils import from_currsys
from . import logger
from ..effects import Effect

PLOT = True

class Oversample(Effect):
    """Oversamples obj.hdu.data (and WCS header) once. Leaves boolean flag to signal Downsample to undo."""
    z_order: ClassVar[tuple[int, ...]] = (601,)  # just before SpacecraftPointing

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.meta.update(kwargs)
        params = {
            "flux_accuracy": 1e-6, 
            "psf_oversampling" : 10,
            "oversampling_x": "!SIM.computing.oversampling_x",
            "oversampling_y": "!SIM.computing.oversampling_y",
        }
        self.meta.update(params)
        self.meta = from_currsys(self.meta, self.cmds)
        self.oversampling_x = self.meta["oversampling_x"]
        self.oversampling_y = self.meta["oversampling_y"]

        # oversampling_x should be a multiple of 2, and both should be divisble by 10 if the UVEXSlitPSF effect is applied
        if self.oversampling_x % 2 != 0:
            logger.warning("Oversampling_x must be divisible by 2 for use with the UVEX Slit Mask effect.")
        if self.meta["psf_oversampling"] % self.oversampling_x != 0:
            logger.warning("The oversampling_x factor must divide into the oversampling factor of the UVEX PSFs if any of the UVEX PSF effects are used.")
        if self.meta["psf_oversampling"] % self.oversampling_y != 0:
            logger.warning("The oversampling_y factor must divide into the oversampling factor of the UVEX PSFs if any of the UVEX PSF effects are used.")

    def _oversample(self, img):
        """
        Oversample an input image in either the x or y direction, or both. 
        The oversampling factor(s) are set in the relevant mode yaml (e.g. UVIM_LSS.yaml).
        """
        if img.ndim == 3: # We are in units of flux density. TODO: Add BUNIT check. Call obj instead of obj.hdu.data as the argument.
            if self.oversampling_y == 1 and self.oversampling_x != 1:
                new_img = np.repeat(img, self.oversampling_x, axis=2) # x only
            elif self.oversampling_y != 1 and self.oversampling_x == 1:
                new_img = np.repeat(img, self.oversampling_y, axis=1) # y only
                logger.warning("Because of the orientation of the UVEX slit, it is recommended at this step to oversample the image in the x direction," \
                "either in addition to or instead of oversampling in the y direction.")
            elif self.oversampling_x != 1 and self.oversampling_y != 1:
                new_img = np.repeat(np.repeat(img, self.oversampling_y, axis=1), self.oversampling_x, axis=2)
            else:
                new_img = img
        elif img.ndim == 2: # We are units of electron counts. TODO: Add BUNIT check.
            if self.oversampling_y == 1 and self.oversampling_x != 1:
                oversampled_image = np.repeat(img, self.oversampling_x, axis=1) # x only
                new_img = oversampled_image / self.oversampling_x
            elif self.oversampling_y != 1 and self.oversampling_x == 1:
                oversampled_image = np.repeat(img, self.oversampling_y, axis=0) # y only
                new_img = oversampled_image / self.oversampling_y
                logger.warning("Because of the orientation of the UVEX slit, it is recommended at this step to oversample the image in the x direction," \
                "either in addition to or instead of oversampling in the y direction.")
            elif self.oversampling_x != 1 and self.oversampling_y != 1:
                oversampled_image = np.repeat(np.repeat(img, self.oversampling_y, axis=0), self.oversampling_x, axis=1)
                new_img = oversampled_image / (self.oversampling_x * self.oversampling_y)
            else:
                new_img = img

        # check flux conservation after oversampling + normalization
        img_sum = img.sum()
        new_sum = new_img.sum()
        if img.ndim == 3:
            new_sum /= (self.oversampling_x * self.oversampling_y)
        if np.isfinite(img_sum) and img_sum != 0:
            rel_diff = np.abs(img_sum - new_sum) / np.abs(img_sum)
            if rel_diff > self.meta["flux_accuracy"]:
                logger.warning("Flux is not conserved by oversampling: difference is %.2f%%", rel_diff * 100)
        return new_img

    def apply_to(self, obj):
        
        if getattr(obj, "_oversampled", False):
            logger.warning("Oversample called on an already-oversampled image; skipping.")
            return obj

        img = self._oversample(obj.hdu.data.astype(float))
        obj.hdu.data = img

        if img.ndim == 3:
            obj.hdu.header["CDELT1"] /= self.oversampling_x
            obj.hdu.header["CRPIX1"] = (obj.hdu.header["CRPIX1"] - 0.5) * self.oversampling_x + 0.5
            obj.hdu.header["CDELT2"] /= self.oversampling_y
            obj.hdu.header["CRPIX2"] = (obj.hdu.header["CRPIX2"] - 0.5) * self.oversampling_y + 0.5

        obj.hdu.header["NAXIS1"] *= self.oversampling_x
        obj.hdu.header["NAXIS2"] *= self.oversampling_y

        obj.hdu.header["CDELT1D"] /= self.oversampling_x
        obj.hdu.header["CRPIX1D"] = (obj.hdu.header["CRPIX1D"] - 0.5) * self.oversampling_x + 0.5
        obj.hdu.header["CDELT2D"] /= self.oversampling_y
        obj.hdu.header["CRPIX2D"] = (obj.hdu.header["CRPIX2D"] - 0.5) * self.oversampling_y + 0.5

        obj._oversampled = True
        obj._oversample_info = dict(oversampling_x=self.oversampling_x,
                                    oversampling_y=self.oversampling_y)
        return obj


class Downsample(Effect):
    """
    Reverses a matching Oversample. Place after the last effect that needs fine resolution.
    
    It's recommended that this effect be applied after mapping a 3D spectral cube to an image plane.
    Otherwise, depending on the placement of the source or the y-axis cropping of a spectral image,
    we might have some flux loss that depends on whether the source falls on a pixel center or edge.
    This appears to be independent of the oversampling factor.
    """
    z_order: ClassVar[tuple[int, ...]] = (698,)  

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.meta.update(kwargs)
        params = {
            "flux_accuracy": 1e-4,
        }
        self.meta.update(params)

    def _downsample(self, img, oversampling_x=1, oversampling_y=1):
        """
        Downsample an input image in either the x or y direction, or both. 
        The oversampling factor(s) are set in the relevant mode yaml (e.g. UVIM_LSS.yaml).
        """
        if img.ndim == 3:
            n_lambda, n_y, n_x = img.shape
            if oversampling_y == 1 and oversampling_x != 1:
                new_n_x = n_x // oversampling_x
                downsampled_image = img.reshape(n_lambda, n_y, new_n_x, oversampling_x).sum(axis=3)
            elif oversampling_y != 1 and oversampling_x == 1:
                new_n_y = n_y // oversampling_y
                downsampled_image = img.reshape(n_lambda, new_n_y, oversampling_y, n_x).sum(axis=2)
            elif oversampling_y != 1 and oversampling_x != 1:
                new_n_y = n_y // oversampling_y
                new_n_x = n_x // oversampling_x
                downsampled_image = img.reshape(n_lambda, new_n_y, oversampling_y, new_n_x, oversampling_x).sum(axis=(2,4))
            else:
                downsampled_image = img
        elif img.ndim == 2:
            n_y, n_x = img.shape
            pad_y = (-n_y) % oversampling_y
            pad_x = (-n_x) % oversampling_x
            if pad_y or pad_x:
                img = np.pad(img, ((0, pad_y), (0, pad_x)), mode="constant", constant_values=0.0)
                n_y, n_x = img.shape

            if oversampling_y == 1 and oversampling_x != 1:
                new_n_x = n_x // oversampling_x
                downsampled_image = img.reshape(n_y, new_n_x, oversampling_x).sum(axis=2)
            elif oversampling_y != 1 and oversampling_x == 1:
                new_n_y = n_y // oversampling_y
                downsampled_image = img.reshape(new_n_y, oversampling_y, n_x).sum(axis=1)
            elif oversampling_y != 1 and oversampling_x != 1:
                new_n_y = n_y // oversampling_y
                new_n_x = n_x // oversampling_x
                downsampled_image = img.reshape(new_n_y, oversampling_y, new_n_x, oversampling_x).sum(axis=(1,3))
            else:
                downsampled_image = img
        # check flux conservation after downsampling
        img_sum = img.sum()
        down_sum = downsampled_image.sum()
        if np.isfinite(img_sum) and img_sum != 0:
            rel_diff = np.abs(img_sum - down_sum) / np.abs(img_sum)
            if rel_diff > self.meta["flux_accuracy"]:
                logger.warning("Flux is not conserved by downsampling: difference is %.2f%%", rel_diff * 100)
        new_img = downsampled_image
        return new_img

    def apply_to(self, obj):
        
        os_info = getattr(obj, "_oversample_info", None)
        if os_info is None:
            logger.warning("Downsample called but the image is not oversampled; skipping.")
            return obj

        ox, oy = os_info["oversampling_x"], os_info["oversampling_y"]
        img = self._downsample(obj.hdu.data, oversampling_x=ox, oversampling_y=oy)
        
        obj.hdu.data = img 
        
        if img.ndim == 3:
            obj.hdu.header["CDELT1"] *= ox
            obj.hdu.header["CRPIX1"] = (obj.hdu.header["CRPIX1"] - 0.5) / ox + 0.5
            obj.hdu.header["CDELT2"] *= oy
            obj.hdu.header["CRPIX2"] = (obj.hdu.header["CRPIX2"] - 0.5) / oy + 0.5

        obj.hdu.header["NAXIS1"] /= ox
        obj.hdu.header["NAXIS2"] /= oy

        obj.hdu.header["CDELT1D"] *= ox
        obj.hdu.header["CRPIX1D"] = (obj.hdu.header["CRPIX1D"] - 0.5) / ox + 0.5
        obj.hdu.header["CDELT2D"] *= oy
        obj.hdu.header["CRPIX2D"] = (obj.hdu.header["CRPIX2D"] - 0.5) / oy + 0.5

        if PLOT:
            import matplotlib.pyplot as plt
            plt.title("Image slice after downsampling")
            if img.ndim == 3:
                plt.imshow(img[img.shape[0] // 2, :, :])
            elif img.ndim == 2:
                plt.imshow(img)
            plt.show()
        
        del obj._oversampled, obj._oversample_info
        return obj
