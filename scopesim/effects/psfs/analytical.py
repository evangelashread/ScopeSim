# -*- coding: utf-8 -*-
"""Contains simple Vibration, NCPA, Seeing and Diffraction PSFs."""

from typing import ClassVar

import numpy as np
from astropy import units as u
from astropy.wcs import WCS
from astropy.convolution import Gaussian2DKernel
from scipy.signal import fftconvolve
from scipy.ndimage import rotate
from tqdm import tqdm

from ...optics import ImagePlane
from ...optics.fov import FieldOfView
from ...optics.fov_volume_list import FovVolumeList
from ...utils import (from_currsys, quantify, quantity_from_table,
                      figure_factory, check_keys, get_logger)
from . import PSF, PoorMansFOV
from .psf_base import get_bkg_level, rotational_blur

logger = get_logger(__name__)

class AnalyticalPSF(PSF):
    """Base class for analytical PSFs."""

    z_order: ClassVar[tuple[int, ...]] = (41, 641)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.convolution_classes = FieldOfView


class Vibration(AnalyticalPSF):
    """Creates a wavelength independent kernel image."""

    required_keys = {"fwhm", "pixel_scale"}
    z_order: ClassVar[tuple[int, ...]] = (244, 744)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.meta["width_n_fwhms"] = 4
        self.convolution_classes = ImagePlane

        check_keys(self.meta, self.required_keys, action="error")
        self.kernel = None

    def get_kernel(self, obj):
        if self.kernel is not None:
            return self.kernel

        from_currsys(self.meta, self.cmds)
        fwhm_pix = self.meta["fwhm"] / self.meta["pixel_scale"]
        sigma = fwhm_pix / 2.35
        width = max(1, int(fwhm_pix * self.meta["width_n_fwhms"]))
        self.kernel = Gaussian2DKernel(sigma, x_size=width, y_size=width,
                                       mode="center").array
        self.kernel /= np.sum(self.kernel)

        return self.kernel.astype(float)


class NonCommonPathAberration(AnalyticalPSF):
    """
    TBA.

    Needed: pixel_scale
    Accepted: kernel_width, strehl_drift
    """

    required_keys = {"pixel_scale"}
    z_order: ClassVar[tuple[int, ...]] = (241, 641)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.meta["kernel_width"] = None
        self.meta["strehl_drift"] = 0.02
        self.meta["wave_min"] = "!SIM.spectral.wave_min"
        self.meta["wave_max"] = "!SIM.spectral.wave_max"

        self._total_wfe = None

        self.valid_waverange = [0.1 * u.um, 0.2 * u.um]

        self.convolution_classes = FieldOfView
        check_keys(self.meta, self.required_keys, action="error")

    def get_kernel(self, obj):
        waves = obj.meta["wave_min"], obj.meta["wave_max"]

        old_waves = self.valid_waverange
        wave_mid_old = 0.5 * (old_waves[0] + old_waves[1])
        wave_mid_new = 0.5 * (waves[0] + waves[1])
        strehl_old = wfe2strehl(wfe=self.total_wfe, wave=wave_mid_old)
        strehl_new = wfe2strehl(wfe=self.total_wfe, wave=wave_mid_new)

        if np.abs(1 - strehl_old / strehl_new) > self.meta["strehl_drift"]:
            self.valid_waverange = waves
            self.kernel = wfe2gauss(wfe=self.total_wfe, wave=wave_mid_new,
                                    width=self.meta["kernel_width"])
            self.kernel /= np.sum(self.kernel)

        return self.kernel

    def _get_total_wfe_from_table(self):
        wfes = quantity_from_table("wfe_rms", self.table, "um")
        n_surfs = self.table["n_surfaces"]
        return np.sum(n_surfs * wfes**2)**0.5

    @property
    def total_wfe(self):
        if self._total_wfe is not None:
            return self._total_wfe

        if self.table is not None:
            self._total_wfe = self._get_total_wfe_from_table()
        else:
            self._total_wfe = 0

        return self._total_wfe

    def plot(self):
        fig, axes = figure_factory()

        wave_min, wave_max = from_currsys([self.meta["wave_min"],
                                           self.meta["wave_max"]], self.cmds)
        waves = np.linspace(wave_min, wave_max, 1001) * u.um
        wfe = self.total_wfe
        strehl = wfe2strehl(wfe=wfe, wave=waves)

        axes.plot(waves, strehl)
        axes.set_xlabel(f"Wavelength [{waves.unit}]")
        axes.set_ylabel(f"Strehl Ratio \n[Total WFE = {wfe}]")

        return fig


class SeeingPSF(AnalyticalPSF):
    """
    Currently only returns gaussian kernel with a ``fwhm`` [arcsec].

    Parameters
    ----------
    fwhm : flaot
        [arcsec]

    """

    z_order: ClassVar[tuple[int, ...]] = (242, 642)

    def __init__(self, fwhm=1.5, **kwargs):
        super().__init__(**kwargs)

        self.meta["fwhm"] = fwhm

    def get_kernel(self, fov):
        # called by .apply_to() from the base PSF class

        pixel_scale = fov.header["CDELT1"] * u.deg.to(u.arcsec)
        pixel_scale = quantify(pixel_scale, u.arcsec)

        # add in the conversion to fwhm from seeing and wavelength here
        fwhm = from_currsys(self.meta["fwhm"], self.cmds) * u.arcsec / pixel_scale

        sigma = fwhm.value / 2.35
        kernel = Gaussian2DKernel(sigma, mode="center").array
        kernel /= np.sum(kernel)

        return kernel

    def plot(self):
        pixel_scale = from_currsys("!INST.pixel_scale", self.cmds)
        spec_dict = from_currsys("!SIM.spectral", self.cmds)
        return super().plot(PoorMansFOV(pixel_scale, spec_dict))
    
class SpacecraftPointing(AnalyticalPSF):
    z_order: ClassVar[tuple[int, ...]] = (202, 602)

    def __init__(self, fwhm=1.5, **kwargs):
        super().__init__(**kwargs)
        self.meta.update(kwargs)
        self.meta["fwhm"] = fwhm
        params = {
            "flux_accuracy": 1e-4,
            "crop_y": "!SIM.computing.crop_y",
        }
        self.oversampling_x = self.meta.get("oversampling_x", 1)
        self.oversampling_y = self.meta.get("oversampling_y", 1)
        if self.oversampling_x != 1 or self.oversampling_y != 1:
            self.oversample_flag = True
        else:
            self.oversample_flag = False
        self.convolution_classes = FieldOfView
        self.meta.update(params)

    def get_kernel(self, fov):
        pixel_scale_x = np.abs(fov.header["CDELT1"]) * u.deg.to(u.arcsec) * u.arcsec
        pixel_scale_y = np.abs(fov.header["CDELT2"]) * u.deg.to(u.arcsec) * u.arcsec
        pixel_scale_x /= self.oversampling_x
        pixel_scale_y /= self.oversampling_y

        fwhm = from_currsys(self.meta["fwhm"], self.cmds) * u.Unit(self.meta["fwhm_unit"])
        fwhm = fwhm.to(u.arcsec)

        sigma_x = (fwhm.value / pixel_scale_x.value) / (2 * np.sqrt(2 * np.log(2)))
        sigma_y = (fwhm.value / pixel_scale_y.value) / (2 * np.sqrt(2 * np.log(2)))

        half_x = int(10 * sigma_x)
        half_y = int(10 * sigma_y)
        x = np.arange(-half_x, half_x + 1)
        y = np.arange(-half_y, half_y + 1)
        xx, yy = np.meshgrid(x, y)

        kernel = gauss2d(x=xx, y=yy, mx=0, my=0, sx=sigma_x, sy=sigma_y)
        kernel /= np.sum(kernel)

        return kernel

    def _oversample(self, img):
        """
        Oversample an input image in either the x or y direction, or both. 
        The oversampling factor(s) are set in UVEX.yaml.
        """
        assert img.ndim == 3, "SpacecraftPointing applies to 3D data cubes only." # not mapped to detector plane yet

        if self.oversampling_y == 1 and self.oversampling_x != 1:
            oversampled_image = np.repeat(img, self.oversampling_x, axis=2) # x only
            new_img = oversampled_image / self.oversampling_x
        elif self.oversampling_y != 1 and self.oversampling_x == 1:
            oversampled_image = np.repeat(img, self.oversampling_y, axis=1) # y only
            new_img = oversampled_image / self.oversampling_y
            logger.warning("Because of the orientation of the slit, it is recommended at this step to oversample the image in the x direction," \
            "either in addition to or instead of oversampling in the y direction.")
        elif self.oversampling_x != 1 and self.oversampling_y != 1:
            oversampled_image = np.repeat(np.repeat(img, self.oversampling_y, axis=1), self.oversampling_x, axis=2)
            new_img = oversampled_image / (self.oversampling_x * self.oversampling_y)

        # check flux conservation after oversampling + normalization
        img_sum = img.sum()
        new_sum = new_img.sum()
        if np.isfinite(img_sum) and img_sum != 0:
            rel_diff = np.abs(img_sum - new_sum) / np.abs(img_sum)
            if rel_diff > self.meta["flux_accuracy"]:
                logger.warning("Flux is not conserved by oversampling: difference is %.2f%%", rel_diff * 100)
        return new_img
        
    def _downsample(self, img):
        """
        Downsample an input image in either the x or y direction, or both. 
        The oversampling factor(s) are set in UVEX.yaml.
        """
        assert img.ndim == 3, "SpacecraftPointing applies to 3D data cubes only." # not mapped to detector plane yet
        n_lambda, n_y, n_x = img.shape
        if self.oversampling_y == 1 and self.oversampling_x != 1:
            new_n_x = n_x // self.oversampling_x
            downsampled_image = img.reshape(n_lambda, n_y, new_n_x, self.oversampling_x).sum(axis=3)
        elif self.oversampling_y != 1 and self.oversampling_x == 1:
            new_n_y = n_y // self.oversampling_y
            downsampled_image = img.reshape(n_lambda, new_n_y, self.oversampling_y, n_x).sum(axis=2)
        elif self.oversampling_y != 1 and self.oversampling_x != 1:
            new_n_y = n_y // self.oversampling_y
            new_n_x = n_x // self.oversampling_x
            downsampled_image = img.reshape(n_lambda, new_n_y, self.oversampling_y, new_n_x, self.oversampling_x).sum(axis=(2,4))

        # check flux conservation after downsampling
        img_sum = img.sum()
        down_sum = downsampled_image.sum()
        if np.isfinite(img_sum) and img_sum != 0:
            rel_diff = np.abs(img_sum - down_sum) / np.abs(img_sum)
            if rel_diff > self.meta["flux_accuracy"]:
                logger.warning("Flux is not conserved by downsampling: difference is %.2f%%", rel_diff * 100)
        new_img = downsampled_image
        return new_img

    def apply_to(self, obj, **kwargs):
        """Apply the PSF."""
        # 1. During setup of the FieldOfViews
        if isinstance(obj, FovVolumeList) and self._waveset is not None:
            logger.debug("Executing %s, FoV setup", self.meta['name'])
            waveset = self._waveset
            if len(waveset) != 0:
                waveset_edges = 0.5 * (waveset[:-1] + waveset[1:])
                obj.split("wave", quantify(waveset_edges, u.um).value)

        # 2. During observe: convolution
        elif isinstance(obj, self.convolution_classes):
            logger.debug("Executing %s, convolution", self.meta['name'])
            if ((hasattr(obj, "fields") and len(obj.fields) > 0) or
                    (obj.hdu is not None)):
                
                kernel = self.get_kernel(obj).astype(float)
                master_img = obj.hdu.data.copy()

                # apply rotational blur for field-tracking observations
                rot_blur_angle = self.meta["rotational_blur_angle"]
                if abs(rot_blur_angle << u.deg) > 0*u.deg:
                    # makes a copy of kernel
                    kernel = rotational_blur(kernel, rot_blur_angle)

                if self.oversample_flag:
                    # Only oversample around the region of interest if desired
                    crop_y = from_currsys(self.meta["crop_y"], self.cmds)
                    if crop_y is not None:
                        crop_unit = u.Unit(from_currsys("!SIM.computing.crop_unit", self.cmds))
                        crop_y = crop_y * crop_unit
                        crop_y = crop_y.to(u.arcsec)
                        x_src, y_src = [], []
                        for field in obj.fields:
                            if field.field is not None:
                                x_src.extend(field.field["x"].value) # in arcsec
                                y_src.extend(field.field["y"].value)
                        
                        nlam, ny, nx = obj.hdu.data.shape
                        wcs = WCS(obj.hdu.header)
                        ys, xs = np.mgrid[0:ny, 0:nx]
                        lambdas = np.zeros_like(xs, dtype=float) # just use first wavelength slice (mask is same for all wavelenght slices)
                        xfld, yfld, _ = wcs.pixel_to_world(xs, ys, lambdas) # deg
                        
                        y_src_arr = np.array(y_src)
                        y_src_max = np.max(y_src_arr) + crop_y.value
                        y_src_min = np.min(y_src_arr) - crop_y.value
                        mask = (yfld.value >= y_src_min / 3600.) & (yfld.value <= y_src_max / 3600.) # in deg
                        y_rows = np.where(mask.any(axis=1))[0]
                        y_lo, y_hi = int(y_rows[0]), int(y_rows[-1]) + 1
                        obj.hdu.header["CRPIX2"] -= y_lo
                        _image = obj.hdu.data[:,y_lo:y_hi,:].astype(float)
                    else:
                        y_lo, y_hi = 0, obj.hdu.data.shape[1]
                        _image = obj.hdu.data.astype(float)
                    image = self._oversample(_image)
                    # Need to update the header accordingly
                    obj.hdu.header["CDELT1"] /= self.oversampling_x
                    obj.hdu.header["CRPIX1"] = (obj.hdu.header["CRPIX1"] - 0.5) * self.oversampling_x + 0.5
                    obj.hdu.header["CDELT2"] /= self.oversampling_y
                    obj.hdu.header["CRPIX2"] = (obj.hdu.header["CRPIX2"] - 0.5) * self.oversampling_y + 0.5
                else:
                    image = obj.hdu.data.astype(float)

                # do the convolution
                logger.debug("PSF convolution start")
                n_lam, n_y, n_x = image.shape
                new_image = np.zeros_like(image, dtype=float)
                bkg_level = get_bkg_level(image, self.meta["bkg_width"])

                with tqdm(total=n_lam, desc=" SpacecraftPointing effect convolution") as pbar:
                    for i in range(n_lam):
                        plane = image[i] # (ny, nx) with x already oversampled
                        bkg = bkg_level[i]
                        new_image[i] = fftconvolve(plane - bkg, kernel, mode="same") + bkg
                        pbar.update(1)

                if self.oversample_flag:
                    if crop_y is not None:
                        master_img[:,y_lo:y_hi,:] = self._downsample(new_image)
                        obj.hdu.data = master_img
                    else:
                        obj.hdu.data = self._downsample(new_image)
                    obj.hdu.header["CDELT1"] *= self.oversampling_x
                    obj.hdu.header["CRPIX1"] = (obj.hdu.header["CRPIX1"] - 0.5) / self.oversampling_x + 0.5
                    obj.hdu.header["CDELT2"] *= self.oversampling_y
                    obj.hdu.header["CRPIX2"] = (obj.hdu.header["CRPIX2"] - 0.5) / self.oversampling_y + 0.5
                    if crop_y is not None:
                        obj.hdu.header["CRPIX2"] += y_lo
                else:
                    obj.hdu.data = new_image

                logger.debug("PSF convolution done")

        return obj

class GaussianDiffractionPSF(AnalyticalPSF):
    z_order: ClassVar[tuple[int, ...]] = (242, 642)

    def __init__(self, diameter, **kwargs):
        super().__init__(**kwargs)
        self.meta["diameter"] = diameter

    def update(self, **kwargs):
        if "diameter" in kwargs:
            self.meta["diameter"] = kwargs["diameter"]

    def get_kernel(self, fov):
        # called by .apply_to() from the base PSF class

        pixel_scale = fov.header["CDELT1"] * u.deg.to(u.arcsec)
        pixel_scale = quantify(pixel_scale, u.arcsec)

        wave = 0.5 * (fov.meta["wave_max"] + fov.meta["wave_min"])

        wave = quantify(wave, u.um)
        diameter = quantify(self.meta["diameter"], u.m).to(u.um)
        fwhm = 1.22 * (wave / diameter) * u.rad.to(u.arcsec) / pixel_scale

        sigma = fwhm.value / 2.35
        kernel = Gaussian2DKernel(sigma, mode="center").array
        kernel /= np.sum(kernel)

        return kernel

    def plot(self):
        pixel_scale = from_currsys("!INST.pixel_scale", self.cmds)
        spec_dict = from_currsys("!SIM.spectral", self.cmds)
        return super().plot(PoorMansFOV(pixel_scale, spec_dict))


def wfe2gauss(wfe, wave, width=None):
    strehl = wfe2strehl(wfe, wave)
    sigma = _strehl2sigma(strehl)
    if width is None:
        width = int(np.ceil(8 * sigma))
        width += (width + 1) % 2
    gauss = _sigma2gauss(sigma, x_size=width, y_size=width)

    return gauss


def wfe2strehl(wfe, wave):
    wave = quantify(wave, u.um)
    wfe = quantify(wfe, u.um)
    x = 2 * 3.1415926526 * wfe / wave
    strehl = np.exp(-x**2)
    return strehl


def _strehl2sigma(strehl):
    amplitudes = [0.00465, 0.00480, 0.00506, 0.00553, 0.00637, 0.00793,
                  0.01092, 0.01669, 0.02736, 0.04584, 0.07656, 0.12639,
                  0.20474, 0.32156, 0.48097, 0.66895, 0.84376, 0.95514,
                  0.99437, 0.99982, 0.99999]
    sigmas = [19.9526, 15.3108, 11.7489, 9.01571, 6.91830, 5.30884, 4.07380,
              3.12607, 2.39883, 1.84077, 1.41253, 1.08392, 0.83176, 0.63826,
              0.48977, 0.37583, 0.28840, 0.22130, 0.16982, 0.13031, 0.1]
    sigma = np.interp(strehl, amplitudes, sigmas)
    return sigma


def _sigma2gauss(sigma, x_size=15, y_size=15):
    kernel = Gaussian2DKernel(sigma, x_size=x_size, y_size=y_size,
                              mode="oversample").array
    kernel /= np.sum(kernel)
    return kernel

def gauss2d(x=0, y=0, mx=0, my=0, sx=1, sy=1):
    return 1. / (2. * np.pi * sx * sy) * np.exp(-((x - mx)**2. / (2. * sx**2.) + (y - my)**2. / (2. * sy**2.)))