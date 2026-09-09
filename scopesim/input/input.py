# -*- coding: utf-8 -*-
"""Module for generating ScopeSim source object(s) for UVEX from user input
   Taking into account telescope orientation, background, etc."""
import astropy
import numpy as np
import astropy.table
import synphot.units
from astropy.time import Time
from astropy.coordinates import SkyCoord, get_sun, UnitSphericalRepresentation
from synphot import SourceSpectrum, ConstFlux1D
from astropy.io import fits
import astropy.units as u
import warnings


class UVEXInput:
    '''
    Three types of data can be loaded into the UVEXInput class, each
    in the form of a table.
    
    1) Point sources with spectra, given in point_sources_from_spectra.
    Each source has a position (ra, dec) and a reference to a spectrum (ref).
    This reference is a string which matches a key in the spectra dictionary.
    Each point source also comes with a scale factor which multiplies its referenced spectrum.

    2) Point sources with magnitudes, give in point_sources_from_magnitude.
    Their position (ra, dec) as well as their magnitude (in ABmag) are included in this table.
    A constant spectrum in magnitudes is assumed.

    3) FITS files containing images, given in patches.
    This table contains position information (ra, dec) for each FITS image, paths to the images (path)
    and also spectral references (ref) which are strings that match spectra dictionary keys.
    We assume that the data is in the primary HDU, and header metadata is ignored.

    Object attributes after validation functions:

        1. observation_coordinates: This is a SkyCoord object for the observation coordinates
        2. observation_time: A astropy Time object for the observation time
        3. use_background: bool for if we should generate a background
        4. point_sources_from_spectrum: Astropy QTable with headers: "ra", "dec", "ref", "scale". "ra" and "dec" have units.
        "ref" is a scalar value which matches a key in the spectra dictionary. Scale is a unitless quantity which scales the referenced spectrum.
        scale exists so that multiple point sources can reference the same spectrum up to differing multiplicative factors.
        5. point_sources_from_magnitude: Astropy QTable with headers: "ra", "dec", "mag". "ra" and "dec" and "mag" all have units.
        These point sources should be converted to SourceSpectrum objects using the synphot ConstFlux1D model
        6. patches: Astropy QTable with headers: "ra", "dec", "ref", "path", where "ra" and "dec" have units. Each value of "ref" matches a key in
        spectra. "path" values are paths to valid FITS files. "ra" and "def" give the location of the center of the FITS image.
        All FITS headers are ignored (except the basic ones) and we take the position information from the table "ra" and "dec" values
        7. spectra: A dictionary with string keys that match "ref" values in astropy tables. The values are synphot SourceSpectrum objects.

        TODO: I can add an input for extinction when it is decided how the user will specify it
    '''

    def __init__(self,
                 observation_coordinates: SkyCoord,
                 observation_time: astropy.time.Time,
                 mode: str,
                 use_background: bool,
                 point_sources_from_spectra: astropy.table.QTable = None,
                 point_sources_from_magnitude: astropy.table.QTable = None,
                 patches: astropy.table.QTable = None,
                 spectra: dict[str, SourceSpectrum] = None,
                 diag: bool = False):

        """
        :param observation_coordinates: The SkyCoord object describing where the observation takes place.
        :param observation_time: An astropy.time.Time object describing when the observation takes place.
        :param mode: One of "fuv", "nuv" or "lss""
        :param use_background: A boolean value indicating whether to generate a background from the observation coordinates and time.
        :param point_sources_from_spectra: An astropy QTable of point sources with spectra given in the spectra parameter.
        The columns must be "ra", "dec", "ref" (string reference to the spectrum in spectra), and scale which is a scalar value multiplying the referenced spectrum.
        :param point_sources_from_magnitude: An astropy QTable of point sources generated with the SourceObject Constant spectrum.
        Required columns are "ra", "dec", "magnitude"
        :param spectra: An astropy QTable of spectra with columns "ref" and "spectra".
        Elements in the "ref" column are strings matching those in the point_sources_from_spectrum table.
        Elements in the spectra column are synphot SourceSpectrum objects.
        :param patches: An astropy QTable for fits file inputs. The columns must be "ra", "dec", "ref", "path".
        "ra", "dec" give the center of the image, "ref" is a string matching the "ref" column in the spectra parameter.
        """
        warnings.simplefilter('default',UserWarning)
        
        if not mode in ["fuv", "nuv", "lss"]:
            raise Exception("Valid modes are 'fuv', 'nuv', 'lss'")
        
        self.mode = mode
        self.validate_observation_coordinates(observation_coordinates, observation_time)
        self.observation_coordinates = observation_coordinates
        self.observation_time = observation_time

        # Ensure that the astropy QTables inputted are in the expected format
        self.spectra = spectra
        
        points_table = None
        point_refs = []
        if point_sources_from_spectra is not None:
            points_table = self.table_validation(point_sources_from_spectra, #table
                                               ["ra", "dec", "ref", "scale"], #expected headers
                                               [float, float, str, float], #expected column data types
                                               {0:"coord", 1:"coord"}, #header indexes which correspond to columns that must have units
                                               " point_sources_from_spectrum table", #table name to show in error messages
                                               )
            point_refs = points_table['ref'].tolist()
            
        if point_sources_from_magnitude is not None:
            points_table = self.table_validation(point_sources_from_magnitude,
                                               ["ra", "dec", "mag"],
                                               [float, float, float],
                                               {0:"coord", 1:"coord", 2:"mag"},
                                               " point_sources_from_magnitude table")
            point_refs = points_table['ref'].tolist()

        if patches is not None:
            patches = self.table_validation(patches,
                                               ["ra", "dec", "ref", "path"],
                                               [float, float, str, str],
                                               {0:"coord", 1:"coord"},
                                               " patches table")
            patch_refs = patches['ref'].tolist()
            
            self.validate_fits(patches.columns["path"].data.tolist())
        
            warnings.warn("FITS header information in patch files will not be used. "
                          "Position and spectral information for each FITS image will be taken from the patches table.")
        else:
            patch_refs = []

        self.point_sources = points_table
        self.patches = patches
        
        if self.spectra is not None:
            spectrum_refs = self.validate_spectra()
            # ensure that any reference to a spectra occuring in the patches or point source table corresponds to a spectrum in spectra
            self.cross_check(point_refs, patch_refs, spectrum_refs)
        else:
            raise Exception("No spectral information provided")
              
        # Determine roll from coord and date
        # Sets the self.roll_angle parameter
        self.set_roll()
        
        if diag:
            print("Done initializing UVEXInput object")
            print(f"coordinates: {self.observation_coordinates}")
            print(f"time: {self.observation_time}")
            print(f"roll angle: {self.roll_angle}")
            if self.point_sources is not None:
                print(f"{len(self.point_sources)} point sources with spectra")
            if self.patches is not None:
                print(f"{len(self.patches)} 2D patches with spectra")
            if self.spectra is not None:
                print(f"spectra: {self.spectra.keys()}")

            print(self.patches)
        
        # Create background
        if use_background == True:
            # Create background here
            sky_background = np.array([])
            
        

    def table_validation(self, table: astropy.table.QTable,
                         headers: list[str],
                         types: list[type],
                         required_units: dict[int, str],
                         debug_name: str):
        """
        Checks:
        1. table is None or an Astropy QTable with no masking and units
        2. table headers match "headers"
        3. types of elements in columns are "types"
        4. headers specified by headers_with_units have an associated unit

        :param table: AstropyQTable
        :param headers: list of header strings such as "ra" or "ref" that the table should have
        :param types: list of types such as "str" or "float" that the table columns should be
        :param required_units: dictionary of indices with corresponding valid units
        :param debug_name: name of the table as will show up in error messages
        :return: list of values from ref column
        """
        if not isinstance(table, astropy.table.QTable):
            raise TypeError(f"{debug_name} must be an astropy QTable")

        if table.masked:
            raise Exception(f"Masked Tables not supported, {debug_name} has attribute masked = True")

        if not table.colnames == headers:
            raise Exception(
                f"{debug_name} should have {headers} as column headers, it currently has {table.colnames} as headers")

        if len(required_units) > 0:
            try:
                units = table.units
            except AttributeError:
                raise AttributeError(f"{debug_name}  QTable does not have unit attribute and this quantity needs units")
            if isinstance(units, list):
                #QTable units can be either list or dictionary
                units = dict(zip(table.colnames, units))

        for header_index in required_units.keys():
            try:
                unit = units[headers[header_index]]
            except KeyError:
                print(table)
                raise KeyError(f"{debug_name} QTable must have unit specified for {headers[header_index]} columns, currently the units are {units}")

            if required_units[header_index] == "mag":
                correct_units = False
                if (unit == astropy.units.ABmag) or (unit == astropy.units.STmag):
                    correct_units = True
                elif unit == astropy.units.mag:
                    unit = astropy.units.ABmag
                    correct_units = True
                    warnings.warn("'mag' units were provided and values will be converted to AB magnitude")
                if not correct_units:
                    raise Exception(f"Invalid unit specified for {headers[header_index]} column")
                
                table['scale'] = 10**(-0.4 * table.columns[headers[header_index]].value)
                
                if self.spectra is not None:
                    warnings.warn(f"{debug_name} provided; spectra will be disregarded")
                
                # Create a constant flux spectrum to be scaled by the magnitude value
                self.spectra = {'0': SourceSpectrum(ConstFlux1D, amplitude=0*unit)}
                table['ref'] = '0'

        for (header, type) in zip(headers, types):
            column = table.columns[header]
            for (i, element) in enumerate(column.data.tolist()):
                if not isinstance(element, type):
                    raise Exception(f"Element {i} in column {header} of {debug_name} should have type {type} instead of {type(element)}")
                if not element:
                    raise Exception(f"Invalid element {i} in column {header}")

        return table

    @staticmethod
    def validate_fits(fits_paths: list[str]):
        """
        Checks:
        1. Each path opens as a fits file
        :param fits_paths: list of paths to FITS files
        """
        for path in fits_paths:
            try:
                with fits.open(path) as hdul:
                    hdul = fits.open(path)
                    hdul.verify()
            except:
                raise Exception(f"{path} is not a valid fits file")
        
        return True

    @staticmethod
    def validate_observation_coordinates(observation_coordinates: SkyCoord, observation_time: Time):
        """
        Checks:
        1. observation_coordinates is an astropy SkyCoord type
        2. observation_time is an astropy Time object
        3. if the obstime of the observation_coordinates has been set then it matches observation_time
        4. observation_time has only one time value
        5. there is exactly one RA and one DEC value (since SkyCoord can be initialized with arrays of coords)
        """
        if not isinstance(observation_coordinates, SkyCoord):
            raise TypeError("observation_coordinates must be a SkyCoord")
        if not isinstance(observation_time, Time):
            raise TypeError("observation_time must be an astropy Time")
        # check if observation coordinates has an obstime (which is optional for SkyCoord)
        if isinstance(observation_coordinates.obstime, Time):
            #if so this time must match observation time
            if not observation_coordinates.obstime == observation_time:
                raise Exception("If observation_coordinates is initialized with an obstime it must match observation_time")
        #ensure there is only one time
        if not observation_time.size == 1:
            raise Exception("Time must have exactly one time value")
        #check that there is exactly one value each for RA and DEC
        if not observation_coordinates.size == 1:
            raise Exception("observation_coordinates must have exactly one RA and one DEC value")

        return True

    def validate_spectra(self):
        """
        Checks:
        1. spectra is None or a dict[str, SourceSpectrum]
        2. References are not empty strings

        :param spectra: dictionary (key: string, value: SourceSpectrum object) containing all spectra referenced in
        input tables
        :return: list of ref values
        """
        #check if spectra is a dictionary
        if not isinstance(self.spectra, dict):
            raise TypeError(
                "spectra must be None or a dictionary of SourceSpectrum objects with the keys equal to ref values in input tables")
        #check each key is a valid string and each value is a SourceSpectrum object
        for key in self.spectra.keys():
            if not isinstance(key, str):
                raise TypeError("keys of spectra dictionary must be strings")
            if not key:
                raise Exception(f"Key must not be an empty string")
            if not isinstance(self.spectra[key], SourceSpectrum):
                raise TypeError("values of spectra must be astropy SourceSpectrum objects")
        return self.spectra.keys()

    @staticmethod
    def cross_check(point_refs: list[str], patch_refs: list[str], spectra_refs: list[str]):
        """
        Checks:
        1. All spectra referenced in the point_sources_from_spectrum table exist as references in spectra.
        2. All spectra referenced in the patches table exist as references in spectra.
        3. Warn for unused spectra.
        :param point_refs: List of spectrum references in point_sources_from_spectrum.
        :param patch_refs: List of spectrum references in patches.
        :param spectrum_refs: List of spectrum keys in spectra
        :return:
        """
        # make sure that all the spectra referenced in the point source table are in the list of spectra
        if not set(point_refs).issubset(set(spectra_refs)):
            raise Exception(
                "Every spectrum reference in point_sources must have a corresponding key in the spectra dictionary \n"
                f"From point_sources table: {set(point_refs)} \n"
                f"From spectra: {set(spectra_refs)} \n")

        # make sure that all the spectra referenced in the FITS table are in the list of spectra
        if not set(patch_refs).issubset(set(spectra_refs)):
            raise Exception(
                "Every spectrum reference in a FITS file must have a corresponding key in the spectra dictionary \n"
                f"From FITS files: {set(patch_refs)} \n"
                f"From spectra: {set(spectra_refs)} \n")

        # warn if some of the spectra given are not used in either the FITS files or the point source table
        if set(spectra_refs).difference(set(point_refs).union(set(patch_refs))):
            warnings.warn(f"Unused spectra: {(set(spectra_refs) - set(patch_refs)) - set(point_refs)}")
    
    def set_roll(self):
        '''
        Calculates the roll angle for a given pointing and date.
        +X is direction along boresight, +Y is Solar panel axis,
        and +Z points to celestial north when roll=0
        
        A more sophisticated approach can be performed using a TLE and the actual
        spacecraft position at any given time. For now, we're just approximating
        using the relative positions of the Sun and the target
        '''
        sun = get_sun(self.observation_time)
        
        v1 = self.observation_coordinates.transform_to(sun.frame).represent_as(UnitSphericalRepresentation)
        v2 = v1.cross(sun.data).represent_as(UnitSphericalRepresentation)
        v3 = v1.cross(v2).represent_as(UnitSphericalRepresentation)
        
        frame = self.observation_coordinates.transform_to(sun.frame).skyoffset_frame()
        # We want y-axis to be 90-deg to the Sun
        y = (SkyCoord(90 * u.deg, 0 * u.deg, frame=frame).transform_to(sun.frame).represent_as(UnitSphericalRepresentation))
        
        self.roll_angle = np.arctan2(v3.dot(y), v2.dot(y)).to(u.deg)
