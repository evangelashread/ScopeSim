"""Placeholder for tests of the UVEX PSF module."""
import pytest
import matplotlib.pyplot as plt
import numpy as np
from scopesim.effects.psfs.psf_base import get_bkg_level
from scopesim.effects.psfs.uvex_psf import LSSDetectorPSF, SlitPSF