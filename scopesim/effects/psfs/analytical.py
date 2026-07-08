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

PLOT = True

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
        self.convolution_classes = FieldOfView
        self.meta.update({"bkg_width": 0.0})

    def get_kernel(self, fov):
        pixel_scale_x = np.abs(fov.hdu.header["CDELT1"]) * u.deg.to(u.arcsec) * u.arcsec
        pixel_scale_y = np.abs(fov.hdu.header["CDELT2"]) * u.deg.to(u.arcsec) * u.arcsec
        
        fwhm = from_currsys(self.meta["fwhm"], self.cmds) * u.Unit(self.meta["fwhm_unit"])
        fwhm = fwhm.to(u.arcsec)

        sigma_x = (fwhm.value / pixel_scale_x.value) / (2 * np.sqrt(2 * np.log(2)))
        sigma_y = (fwhm.value / pixel_scale_y.value) / (2 * np.sqrt(2 * np.log(2)))

        half_x = int(10 * np.ceil(sigma_x))
        half_y = int(10 * np.ceil(sigma_y))
        x = np.arange(-half_x, half_x + 1)
        y = np.arange(-half_y, half_y + 1)
        xx, yy = np.meshgrid(x, y)

        kernel = gauss2d(x=xx, y=yy, mx=0, my=0, sx=sigma_x, sy=sigma_y)
        kernel /= np.sum(kernel)

        return kernel

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

                image = obj.hdu.data.astype(float)

                # do the convolution
                logger.debug("PSF convolution start")
                
                if len(image.shape) == 3:
                    n_lam, n_y, n_x = image.shape
                    new_image = np.zeros_like(image, dtype=float)
                    bkg_level = get_bkg_level(image, self.meta["bkg_width"])

                    with tqdm(total=n_lam, desc=" SpacecraftPointing effect convolution") as pbar:
                        for i in range(n_lam):
                            plane = image[i] # (ny, nx) with x already oversampled
                            bkg = bkg_level[i]
                            new_image[i] = fftconvolve(plane - bkg, kernel, mode="same") + bkg
                            pbar.update(1)
                            
                    if PLOT:
                        import matplotlib.pyplot as plt
                        plt.title("Image slice after SpacecraftPointing")
                        plt.imshow(new_image[new_image.shape[0] // 2,:,:])
                        plt.show()
                elif len(image.shape) == 2:
                    n_y, n_x = image.shape
                    new_image = np.zeros_like(image, dtype=float)
                    bkg_level = get_bkg_level(image, self.meta["bkg_width"])
                    new_image = fftconvolve(image - bkg_level, kernel, mode="same") + bkg_level
                    
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
