# -*- coding: utf-8 -*-
"""Image-plane illumination effects."""

from typing import ClassVar
from collections.abc import Callable, Mapping

import numpy as np
from scipy.ndimage import affine_transform
from astropy import units as u
from astropy.io import fits, ascii
from astropy.wcs import WCS
from astropy.modeling.functional_models import Gaussian2D

from . import Effect
from ..optics.image_plane import ImagePlane
from ..utils import figure_factory
from ..utils import find_file, from_currsys

__all__ = ["Illumination", "FitsIllumination", "gaussian2d", "quadratic_vignetting"]


def gaussian2d(
    shape: tuple[int, int],
    amp: float = 1.0,
    mu: tuple[float, float] = (0.0, 0.0),
    sigma: tuple[float, float] = (2000.0, 2000.0),
    theta: u.Quantity[u.deg] | float = 0.0 * u.deg,
) -> np.ndarray:
    """
    2D elliptical Gaussian to be used for vignetting map.

    .. versionadded:: 0.11.3

    Parameters
    ----------
    shape : tuple[int, int]
        Image shape in pixels (ny, nx).
    amp : float, optional
        Peak throughput. The default is 1.0.
    mu : tuple[float, float], optional
        Offset of the peak center in pixels (x, y) from the image center.
        The default is (0.0, 0.0), i.e. no offset.
    sigma : tuple[float, float], optional
        Gaussian widths in pixels (sx, sy). The default is (2000.0, 2000.0).
    theta : float, optional
        Rotation angle (if float, the angle is interpreted in degrees),
        counterclockwise. The default is 0°.

    Returns
    -------
    np.ndarray
        Vignetting map.

    """
    nx, ny = reversed(shape)
    y, x = np.ogrid[:ny, :nx]
    x = x - nx / 2
    y = y - ny / 2

    model = Gaussian2D(
        amplitude=amp,
        x_mean=mu[0],
        y_mean=mu[1],
        x_stddev=sigma[0],
        y_stddev=sigma[1],
        theta=theta << u.deg,
    )
    return model(x, y)


def quadratic_vignetting(
    shape: tuple[int, int],
    falloff: float = 0.01,
    r_ref: float | None = None,
    mu: tuple[float, float] = (0.0, 0.0),
    stretch: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
) -> np.ndarray:
    """
    Quadratic vignetting pattern with independent stretch factors.

    .. versionadded:: 0.11.3

    Parameters
    ----------
    shape : tuple[int, int]
        Image shape in pixels (ny, nx).
    falloff : float, optional
        Fractional illumination drop at `r_ref`. The default is 0.01 (= 1 %).
    r_ref : float | None, optional
        Reference radius in stretched pixels. If None (the default), use the
        corner distance.
    mu : tuple[float, float], optional
        Offset of the vignetting center in pixels (x, y) from the image center.
        The default is (0.0, 0.0), i.e. no offset.
    stretch : tuple[float, float, float, float], optional
        ``(+x, -x, +y, -y)`` independent scale factors for half-planes
        respectively. All 1.0 gives a circular pattern. A value > 1 widens the
        falloff in that direction (shallower); < 1 narrows it (steeper).
        The default is (1.0, 1.0, 1.0, 1.0).

    Returns
    -------
    np.ndarray
        Vignetting map.

    """
    nx, ny = reversed(shape)

    yy, xx = np.ogrid[:ny, :nx]
    dx = xx - (nx / 2 + mu[0])
    dy = yy - (ny / 2 + mu[1])

    sx = np.where(dx >= 0, stretch[0], stretch[1])
    sy = np.where(dy >= 0, stretch[2], stretch[3])

    r2 = (dx / sx)**2 + (dy / sy)**2

    if r_ref is None:
        r2_ref = r2.max()
    else:
        r2_ref = r_ref**2

    return np.clip(1.0 - falloff * r2 / r2_ref, 0.0, 1.0)


class FitsIllumination(Effect):
    """
    Image-plane illumination map loaded from a FITS file.

    The input FITS illumination map is assumed to represent the full
    detector focal-plane footprint described by ``detector_layout``.

    The detector-coordinate WCS ("D" WCS) of the current ScopeSim
    ImagePlane is used to determine which region of the full illumination
    map corresponds to the current image plane.

    This allows the same illumination FITS file to be used for:

        - the full detector mosaic
        - a single detector
        - an arbitrary subset of detectors

    without stretching the full illumination pattern onto the selected
    detector subset.

    The illumination map is applied by multiplying:

        obj.hdu.data *= illumination_map
    """

    z_order: ClassVar[tuple[int, ...]] = (750,)

    def __init__(
        self,
        filename: str,
        detector_layout: str,
        normalize: bool = False,
        interpolate: bool = True,
        layout_unit: str = "mm",
        **kwargs,
    ) -> None:

        super().__init__(**kwargs)

        self.meta.setdefault("include", "!DET.include_illumination")

        self.meta["filename"] = filename
        self.meta["detector_layout"] = detector_layout
        self.meta["normalize"] = normalize
        self.meta["interpolate"] = interpolate
        self.meta["layout_unit"] = layout_unit

        # Full input illumination map.
        self._source_map = None

        # Full physical detector-layout bounds:
        # (xmin, xmax, ymin, ymax)
        self._reference_bounds = None

        # Illumination map resampled onto the current ImagePlane.
        self._map = None

        # Cache must include both image shape and WCS position.
        self._map_signature = None

    def apply_to(self, obj, **kwargs):
        if not isinstance(obj, ImagePlane):
            return obj

        target_shape = tuple(obj.hdu.data.shape)

        # The map depends not only on the number of pixels, but also on
        # where those pixels lie in the focal plane.
        target_wcs = WCS(obj.hdu.header, key="D")

        wcs_header = target_wcs.to_header(relax=True)

        signature = (
            target_shape,
            wcs_header.tostring(
                sep="\n",
                endcard=False,
                padding=False,
            ),
        )

        if self._map is None or signature != self._map_signature:

            print(
                "FitsIllumination image-plane shape:",
                target_shape,
            )

            self._map = self._make_map(obj)
            self._map_signature = signature

        # Apply vignetting in place.
        obj.hdu.data *= self._map

        return obj

    def _load_source_map(self):
        """
        Load the full-field FITS illumination map.
        """

        if self._source_map is not None:
            return self._source_map

        requested_filename = from_currsys(
            self.meta["filename"],
            self.cmds,
        )

        filename = find_file(requested_filename)

        if filename is None:
            raise FileNotFoundError(
                "Could not locate illumination FITS file: "
                f"{requested_filename}"
            )

        illumination_map = fits.getdata(filename).astype(
            np.float32,
            copy=False,
        )

        if illumination_map.ndim != 2:
            raise ValueError(
                "Illumination FITS file must be 2D, "
                f"got shape {illumination_map.shape}"
            )

        # Normalize the FULL illumination map, rather than the currently
        # selected subsection. Otherwise different detector selections
        # could acquire different normalizations.
        if self.meta["normalize"]:
            maxval = np.nanmax(illumination_map)

            if maxval > 0:
                illumination_map = illumination_map / maxval

        self._source_map = np.asarray(
            illumination_map,
            dtype=np.float32,
        )

        print(
            "FitsIllumination input-map shape:",
            self._source_map.shape,
        )

        return self._source_map

    def _load_reference_bounds(self):
        """
        Determine the physical extent of the full detector mosaic.

        The bounds are derived from the original generated detector-layout
        file, not from the currently active DetectorList table. Therefore,
        restricting ScopeSim to one detector does not change the reference
        focal-plane footprint.
        """

        if self._reference_bounds is not None:
            return self._reference_bounds

        requested_layout = from_currsys(
            self.meta["detector_layout"],
            self.cmds,
        )

        layout_filename = find_file(requested_layout)

        if layout_filename is None:
            raise FileNotFoundError(
                "Could not locate detector layout file: "
                f"{requested_layout}"
            )

        layout = ascii.read(
            layout_filename,
            format="basic",
            guess=False,
        )

        required_columns = {
            "x_cen",
            "y_cen",
            "x_size",
            "y_size",
        }

        column_names = set(layout.colnames)

        missing = required_columns - column_names

        if missing:
            raise ValueError(
                "Detector layout is missing required columns: "
                f"{sorted(missing)}"
            )

        x_cen = np.asarray(layout["x_cen"], dtype=float)
        y_cen = np.asarray(layout["y_cen"], dtype=float)

        x_size = np.asarray(layout["x_size"], dtype=float)
        y_size = np.asarray(layout["y_size"], dtype=float)

        # Account for detector rotation when determining the bounding box.
        # For the current UVEX imaging layout angle = 0 deg, but keeping this
        # here makes the calculation more general.
        if "angle" in column_names:

            theta = np.deg2rad(
                np.asarray(layout["angle"], dtype=float)
            )

            cos_t = np.abs(np.cos(theta))
            sin_t = np.abs(np.sin(theta))

            x_half = 0.5 * (
                cos_t * x_size +
                sin_t * y_size
            )

            y_half = 0.5 * (
                sin_t * x_size +
                cos_t * y_size
            )

        else:

            x_half = 0.5 * x_size
            y_half = 0.5 * y_size

        xmin = np.min(x_cen - x_half)
        xmax = np.max(x_cen + x_half)

        ymin = np.min(y_cen - y_half)
        ymax = np.max(y_cen + y_half)

        self._reference_bounds = (
            float(xmin),
            float(xmax),
            float(ymin),
            float(ymax),
        )

        print(
            "FitsIllumination full focal-plane bounds:",
            self._reference_bounds,
            self.meta["layout_unit"],
        )

        return self._reference_bounds

    def _make_map(self, obj):
        """
        Resample the full-field illumination map onto the current
        ScopeSim ImagePlane using detector-plane coordinates.
        """

        source_map = self._load_source_map()

        xmin, xmax, ymin, ymax = (
            self._load_reference_bounds()
        )

        target_shape = tuple(obj.hdu.data.shape)

        source_ny, source_nx = source_map.shape

        # Physical size represented by one source-map pixel.
        source_dx = (xmax - xmin) / source_nx
        source_dy = (ymax - ymin) / source_ny

        if source_dx <= 0 or source_dy <= 0:
            raise ValueError(
                "Invalid detector-layout bounds."
            )

        # Detector-coordinate WCS of the CURRENT image plane.
        target_wcs = WCS(
            obj.hdu.header,
            key="D",
        )

        if target_wcs.pixel_n_dim != 2:
            raise ValueError(
                "FitsIllumination requires a 2D detector-coordinate WCS."
            )

        # Evaluate the physical position of three detector-image pixels:
        #
        #   (0, 0) : origin
        #   (1, 0) : one output pixel in +x
        #   (0, 1) : one output pixel in +y
        #
        # This gives us the complete linear transformation from output
        # ImagePlane pixels to physical focal-plane coordinates.
        pixel_points = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        )

        world = np.asarray(
            target_wcs.all_pix2world(
                pixel_points,
                0,
            ),
            dtype=float,
        )

        layout_unit = u.Unit(
            self.meta["layout_unit"]
        )

        # WCS units are expected to describe the same physical coordinate
        # system as the detector layout.
        x_unit = u.Unit(
            target_wcs.wcs.cunit[0]
        )

        y_unit = u.Unit(
            target_wcs.wcs.cunit[1]
        )

        world_x = (
            world[:, 0] * x_unit
        ).to_value(layout_unit)

        world_y = (
            world[:, 1] * y_unit
        ).to_value(layout_unit)

        # Physical position of output pixel (0, 0).
        x0 = world_x[0]
        y0 = world_y[0]

        # Physical displacement caused by increasing output x by one pixel.
        dx_world_x = world_x[1] - world_x[0]
        dx_world_y = world_y[1] - world_y[0]

        # Physical displacement caused by increasing output y by one pixel.
        dy_world_x = world_x[2] - world_x[0]
        dy_world_y = world_y[2] - world_y[0]

        # Convert the physical origin to coordinates in the input
        # illumination map.
        #
        # The -0.5 term expresses the assumption that the physical
        # full-focal-plane bounds describe pixel EDGES, while scipy array
        # coordinates refer to pixel CENTERS.
        source_x0 = (
            (x0 - xmin) / source_dx
            - 0.5
        )

        source_y0 = (
            (y0 - ymin) / source_dy
            - 0.5
        )

        # scipy.ndimage.affine_transform uses array coordinate ordering:
        #
        #     (y, x)
        #
        # rather than WCS ordering:
        #
        #     (x, y)
        #
        # The matrix below maps each target ImagePlane pixel to the
        # corresponding position in the full input illumination map.
        matrix = np.array(
            [
                [
                    dy_world_y / source_dy,
                    dx_world_y / source_dy,
                ],
                [
                    dy_world_x / source_dx,
                    dx_world_x / source_dx,
                ],
            ],
            dtype=float,
        )

        offset = np.array(
            [
                source_y0,
                source_x0,
            ],
            dtype=float,
        )

        order = (
            1 if self.meta["interpolate"]
            else 0
        )

        print(
            "FitsIllumination sampling full-field map "
            "onto current image-plane footprint."
        )

        illumination_map = affine_transform(
            source_map,
            matrix=matrix,
            offset=offset,
            output_shape=target_shape,
            order=order,
            mode="nearest",
            prefilter=False,
            output=np.float32,
        )

        if illumination_map.shape != target_shape:
            raise RuntimeError(
                "Interpolation produced an unexpected shape: "
                f"{illumination_map.shape}; "
                f"expected {target_shape}"
            )

        return illumination_map

    def plot(self):
        if self._map is None:
            raise RuntimeError(
                "No illumination map cached — "
                "run a simulation first."
            )

        fig, ax = figure_factory()

        im = ax.imshow(
            self._map,
            origin="lower",
            cmap="gray_r",
        )

        fig.colorbar(
            im,
            ax=ax,
            label="Relative illumination",
        )

        ax.set_title("Fits Illumination")
        ax.set_xlabel("x [px]")
        ax.set_ylabel("y [px]")

        return fig        


class Illumination(Effect):
    """Large-scale illumination variation across the image plane.

    .. versionadded:: 0.11.3

    Parameters
    ----------
    model : callable, optional
        Function ``f(shape, **kwargs) -> ndarray`` returning the
        illumination map. Defaults to :func:`gaussian2d`.
    modelargs : dict, optional
        Keyword arguments forwarded to ``model``. If omitted, the model's
        own defaults are used.

    include : str
        Turn effect on/off from the IRDB
        default.yaml.  Defaults to ``"!DET.include_illumination"``.

    Examples
    --------
    Polynomial vignetting with <1 % falloff (auto r_ref from image shape)

    >>> eff = Illumination(
    ...     model=quadratic_vignetting,
    ...     modelargs={"falloff": 0.01},
    ... )

    Custom model

    >>> def my_model(shape, slope=-0.001):
    >>>     ny, nx = shape[-2], shape[-1]
    >>>     y, x = np.ogrid[:ny, :nx]
    >>>     r = np.sqrt((x - nx / 2)**2 + (y - ny / 2)**2)
    >>>     return np.clip(1 + slope * r, 0, None)
    >>>
    >>> eff = Illumination(model=my_model, modelargs={"slope": -0.0005})

    """

    z_order: ClassVar[tuple[int, ...]] = (750,)

    def __init__(
        self,
        model: Callable = gaussian2d,
        modelargs: Mapping | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.meta.setdefault("include", "!DET.include_illumination")
        self._model = model
        self._modelargs = modelargs or {}
        self._map = None
        self._map_shape = None

    def apply_to(self, obj, **kwargs):
        if not isinstance(obj, ImagePlane):
            return obj

        shape = obj.hdu.data.shape
        print("illumination image-plane shape:", shape)

        if self._map is None or shape != self._map_shape:
            self._map = self._make_map(shape)
            self._map_shape = shape

        obj.hdu.data *= self._map
        return obj

    def _make_map(self, shape):
        illumination_map = self._model(shape, **self._modelargs)
        return illumination_map.astype(np.float32)

    def plot(self):
        """Plot effect."""
        if self._map is None:
            raise RuntimeError(
                "No illumination map cached — run a simulation first."
            )

        fig, ax = figure_factory()
        im = ax.imshow(
            self._map, origin="lower", vmin=0.98, vmax=1., cmap="gray_r",
        )
        fig.colorbar(im, ax=ax, label="Relative illumination")
        ax.set_title("Illumination")
        ax.set_xlabel("x [px]")
        ax.set_ylabel("y [px]")
        return fig
