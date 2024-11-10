# -*- coding: utf-8 -*-
"""Test the PyIRI Main Library functions."""

import numpy as np
import pytest

import PyIRI
import PyIRI.main_library as mlib


def create_test_grid(lon_res=60.0, lat_res=30.0, alt_res=100.0, hr_res=6.0,
                     lon_min=0.0, lon_max=360.0, lat_min=-90.0, lat_max=90.0,
                     alt_min=100.0, alt_max=1000.0):
    """Create a regular 3D grid for a PyIRI run.

    Parameters
    ----------
    lon_res : float
        Longitude resolution in degrees (default=60.0)
    lat_res : float
        Latitude resolution in degrees (default=30.0)
    alt_res : float
        Altitude resolution in km (default=100.0)
    hr_res : float
        Time resolution in hours (default=6.0)
    lon_min : float
        Minimum longitude in degrees (default=0.0)
    lon_max : float
        Maximum longitude in degrees (default=360.0)
    lat_min : float
        Minimum latitude in degrees (default=-90.0)
    lat_max : float
        Maximum latitude in degrees (default=90.0)
    alt_min : float
        Minimum altitude in km (default=100.0)
    alt_max : float
        Maximum altitude in km (default=1000.0)

    Returns
    -------
    lons : np.array
        1D array of longitudes, gridded then flattened
    lats : np.array
        1D array of latitudes, gridded then flattened
    alts : np.array
        1D array of altitudes
    hrs : np.array
        1D array of hours of day

    """
    # Get the range of latitudes and longitudes
    lon_coords = np.arange(lon_min, lon_max + lon_res, lon_res)
    lat_coords = np.arange(lat_min, lat_max + lat_res, lat_res)

    # Grid the longitude and latitude ranges
    lon_grid, lat_grid = np.meshgrid(lon_coords, lat_coords)

    # Flatten the latitude and longitude outputs
    lons = lon_grid.flatten()
    lats = lat_grid.flatten()

    # Prepare the altitude and hour outputs
    alts = np.arange(alt_min, alt_max + alt_res, alt_res)
    hrs = np.arange(0, 24, hr_res)

    return lons, lats, alts, hrs


class TestVTEC(object):
    """Test class for the VTEC function."""

    def setup_method(self):
        """Initialize for every test."""
        self.lons, self.lats, self.alts, self.hrs = create_test_grid()
        self.edp = None
        self.out = None

    def teardown_method(self):
        """Clean up the test environment."""
        del self.lons, self.lats, self.alts, self.hrs, self.out, self.edp

    def set_edp(self):
        """Update the electron density profile."""
        self.out = mlib.IRI_density_1day(2014, 1, 1, self.hrs, self.lons,
                                         self.lats, self.alts, 80,
                                         PyIRI.coeff_dir, ccir_or_ursi=0)
        self.edp = self.out[-1]
        self.out = None
        return

    @pytest.mark.parametrize("min_alt,max_alt", [(0.0, 202000.0),
                                                 (200.0, 300.0)])
    def test_edp_to_vtec_reg_grid(self, min_alt, max_alt):
        """Test VTEC calculation with a regular grid.

        Parameters
        ----------
        min_alt : float
            Minimum integration altitude in km
        max_alt : float
            Maximum integration altitude in km

        """
        # Get the EDP
        self.set_edp()

        # Calculate the VTEC
        self.out = mlib.edp_to_vtec(self.edp, self.alts, min_alt=min_alt,
                                    max_alt=max_alt)

        # Test the dimensions of the output
        assert self.out.shape[0] == self.hrs.shape[0], \
            "unexpected time dimension for VTEC output"
        assert self.out.shape[1] == self.lats.shape[0], \
            "unexpected space dimension for VTEC output"
        assert len(self.out.shape) == 2, "Unexpected number of VTEC dims"

        # Test the VTEC values
        if min_alt == 0.0:
            assert abs(self.out.min() - 0.5987386480712104) <= 1.0e-4, \
                "Unexpected VTEC minimum value"
            assert abs(self.out.max() - 26.679315534279677) <= 1.0e-4, \
                "Unexpected VTEC maximum value"
        else:
            assert abs(self.out.min() - 0.3691955068010374) <= 1.0e-4, \
                "Unexpected VTEC minimum value"
            assert abs(self.out.max() - 16.50155912097381) <= 1.0e-4, \
                "Unexpected VTEC maximum value"
        return

    @pytest.mark.parametrize("min_alt,max_alt", [(0.0, 202000.0),
                                                 (200.0, 300.0)])
    def test_edp_to_vtec_irreg_grid(self, min_alt, max_alt):
        """Test VTEC calculation with a regular grid.

        Parameters
        ----------
        min_alt : float
            Minimum integration altitude in km
        max_alt : float
            Maximum integration altitude in km

        """
        # Update the altitude grid to be irregular
        self.alts = np.array([100.0, 150.0, 210.0, 280.0, 370.0, 470.0, 580.0,
                              700.0, 930.0, 1070.0])

        # Get the EDP
        self.set_edp()

        # Calculate the VTEC
        self.out = mlib.edp_to_vtec(self.edp, self.alts, min_alt=min_alt,
                                    max_alt=max_alt)

        # Test the dimensions of the output
        assert self.out.shape[0] == self.hrs.shape[0], \
            "unexpected time dimension for VTEC output"
        assert self.out.shape[1] == self.lats.shape[0], \
            "unexpected space dimension for VTEC output"
        assert len(self.out.shape) == 2, "Unexpected number of VTEC dims"

        # Test the VTEC values
        if min_alt == 0.0:
            assert abs(self.out.min() - 0.6694397766157791) <= 1.0e-4, \
                "Unexpected VTEC minimum value"
            assert abs(self.out.max() - 28.002154487047168) <= 1.0e-4, \
                "Unexpected VTEC maximum value"
        else:
            assert abs(self.out.min() - 0.30714538984353085) <= 1.0e-4, \
                "Unexpected VTEC minimum value"
            assert abs(self.out.max() - 13.429899945619965) <= 1.0e-4, \
                "Unexpected VTEC maximum value"
        return

    @pytest.mark.parametrize("min_alt,max_alt", [
        (0.0, 90.0), (100000.0, 200000.0), (10.0, 0.0)])
    def test_edp_to_vtec_bad_range(self, min_alt, max_alt):
        """Test VTEC calculation with a regular grid.

        Parameters
        ----------
        min_alt : float
            Minimum integration altitude in km
        max_alt : float
            Maximum integration altitude in km

        """
        # Get the EDP
        self.set_edp()

        # Calculate the VTEC and test the error
        with pytest.raises(ValueError) as verr:
            mlib.edp_to_vtec(self.edp, self.alts, min_alt=min_alt,
                             max_alt=max_alt)

        # Test the error message
        assert str(verr.value).find('Altitude range contains no values') >= 0, \
            "unexpected error message: {:s}".format(str(verr.value))
        return
