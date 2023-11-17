# -*- coding: utf-8 -*-
"""Test the PyIRI IGRF Library functions.

Notes
-----
Functions that rely on the IGRF coefficients are expected have outputs that
change when the IGRF coefficients are updated.

"""

import numpy as np
import pytest

import PyIRI.igrf_library as ilib


class TestIGRFLibUtil(object):
    """Test class for igrf_library functions that do not use coefficients."""

    def setup_method(self):
        """Initialize all tests."""
        # Specify inputs
        self.inc_in = np.arange(-90.0, 90.0, 10.0)
        self.lat_in = np.arange(-90.0, 90.0, 10.0)

        # Specify test outputs
        self.modip_out = np.array([-89.99999971, -73.38240434, -64.42022926,
                                   -55.97131300, -47.42547845, -38.57748519,
                                   -29.36399894, -19.80358396, -9.974826710,
                                   0.00000000, 9.97482671, 19.80358396,
                                   29.36399894, 38.57748519, 47.42547845,
                                   55.97131300, 64.42022926, 73.38240434])
        self.mlat_out = np.array([-90.00000000, -70.57459986, -53.94761127,
                                  -40.89339465, -30.78973303, -22.76047627,
                                  -16.10211375, -10.31410482, -5.038368770,
                                  0.00000000, 5.03836877, 10.31410482,
                                  16.10211375, 22.76047627, 30.78973303,
                                  40.89339465, 53.94761127, 70.57459986])
        self.dec_out = np.array([-135.0, -135.0, -135.0, -135.0, -135.0,
                                 -135.0, -135.0, -135.0, -135.0, 0.0, 45.0,
                                 45.0, 45.0, 45.0, 45.0, 45.0, 45.0, 45.0])
        self.hoz_out = np.array([127.27922061, 113.13708499, 98.99494937,
                                 84.85281374, 70.71067812, 56.56854249,
                                 42.42640687, 28.28427125, 14.14213562, 0.0,
                                 14.14213562, 28.28427125, 42.42640687,
                                 56.56854249,70.71067812, 84.85281374,
                                 98.99494937, 113.13708499])
        self.inc_out = np.array([-35.26438968, -35.26438968, -35.26438968,
                                 -35.26438968, -35.26438968, -35.26438968,
                                 -35.26438968, -35.26438968, -35.26438968,
                                 0.00000000, 35.26438968, 35.26438968,
                                 35.26438968, 35.26438968, 35.26438968,
                                 35.26438968, 35.26438968, 35.26438968])
        self.eff_out = np.array([155.88457268, 138.56406461, 121.24355653,
                                 103.92304845, 86.60254038, 69.28203230,
                                 51.96152423, 34.64101615, 17.32050808,
                                 0.00000000, 17.32050808, 34.64101615,
                                 51.96152423, 69.28203230, 86.60254038,
                                 103.92304845, 121.24355653, 138.56406461])
        return

    def teardown_method(self):
        """Clean up all tests."""
        del self.inc_in, self.lat_in, self.modip_out, self.mlat_out
        del self.dec_out, self.hoz_out, self.inc_out, self.eff_out
        return

    @pytest.mark.parametrize('inc', [45.0, 45, [0.0, 90.0],
                                     np.arange(-90.0, 90.0, 1.0),
                                     np.array([[0.0, 0.0], [90.0, -90.0]])])
    def test_good_inc_verification(self, inc):
        """Test `verify_inclination` using good inputs.

        Parameters
        ----------
        inc : int, float, or array-like
            Magnetic inclination in degrees.

        """
        out = ilib.verify_inclination(inc)

        assert isinstance(out, type(inc))
        np.testing.assert_almost_equal(out, inc, 6)
        return

    @pytest.mark.parametrize('inc', [-90.1, 90.1, [-90.1, 90.1],
                                     np.arange(-90.1, 90.1, 0.1),
                                     np.array([[-90.1, 0.0], [90.1, 90.0]])])
    def test_close_inc_verification(self, inc):
        """Test `verify_inclination` using correctable inputs.

        Parameters
        ----------
        inc : int, float, or array-like
            Magnetic inclination in degrees.

        """
        out = ilib.verify_inclination(inc)

        assert isinstance(out, type(inc))
        np.testing.assert_almost_equal(out, inc, 1)
        return

    @pytest.mark.parametrize('inc', [-90.11, 90.11, [-90.11, 90.11],
                                     np.arange(-90.11, 90.0, 1.0),
                                     np.array([[-90.11, 0.0], [90.1, 90.0]])])
    @pytest.mark.parametrize('function,args', [
        (ilib.verify_inclination, []), (ilib.inc2modip, [10.0]),
        (ilib.inc2magnetic_dip_latitude, [])])
    def test_bad_inc_verification(self, inc, function, args):
        """Test functions that verify inclination using bad inputs.

        Parameters
        ----------
        inc : int, float, or array-like
            Magnetic inclination in degrees.
        function : func
            Function to test
        args : list
            Additional arguments needed by the function

        """

        with pytest.raises(ValueError) as verr:
            function(inc, *args)

        assert str(verr.value).find('unrealistic magnetic inclination') >= 0, \
            "unexpected error message: {:s}".format(str(verr.value))
        return

    @pytest.mark.parametrize('index', [1, slice(None)])
    def test_inc2modip(self, index):
        """Test `inc2modip` using locally specified outputs.

        Parameters
        ----------
        index : int or slice
            Index or slice, to allow testing of float or array-like inputs.

        """

        out = ilib.inc2modip(self.inc_in[index], self.lat_in[index])

        assert isinstance(out, type(self.inc_in[index]))
        np.testing.assert_almost_equal(out, self.modip_out[index], 6)
        return

    @pytest.mark.parametrize('index', [1, slice(None)])
    def test_inc2mag_dip_lat(self, index):
        """Test `inc2magnetic_dip_latitude` using locally specified outputs.

        Parameters
        ----------
        index : int or slice
            Index or slice, to allow testing of float or array-like inputs.

        """
        out = ilib.inc2magnetic_dip_latitude(self.inc_in[index])

        assert isinstance(out, type(self.inc_in[index]))
        np.testing.assert_almost_equal(out, self.mlat_out[index], 6)
        return

    def test_gg_to_geo_to_gg(self):
        """Test conversion from geocentric to geodetic and back."""
        # Define the inputs
        h_in = np.array([100.0, 1000.0, 1500.0, 0.0])
        geo_in = np.array([0.0, 45.0, 90.0, -45.0])

        # Perform the conversions
        rad, theta, sd, cd = ilib.gg_to_geo(h_in, geo_in)
        h_out, geo_out = ilib.geo_to_gg(rad, theta)

        # Test the outputs (hemispheric information is lost)
        np.testing.assert_almost_equal(h_out, h_in, 3)
        np.testing.assert_almost_equal(geo_out, abs(geo_in), 3)
        assert sd.shape == h_in.shape
        assert cd.shape == h_in.shape
        return

    @pytest.mark.parametrize('index', [1, slice(None)])
    def test_xyz2dhif_success(self, index):
        """Test `xyz2dhif` using locally specified outputs.

        Parameters
        ----------
        index : int or slice
            Index or slice, to allow testing of float or array-like inputs.

        """
        x_in = self.inc_in
        y_in = self.inc_in
        z_in = self.inc_in

        dec, hoz, inc, eff = ilib.xyz2dhif(x_in[index], y_in[index],
                                           z_in[index])

        assert isinstance(dec, type(x_in[index]))
        np.testing.assert_almost_equal(dec, self.dec_out[index], 6)
        np.testing.assert_almost_equal(hoz, self.hoz_out[index], 6)
        np.testing.assert_almost_equal(inc, self.inc_out[index], 6)
        np.testing.assert_almost_equal(eff, self.eff_out[index], 6)
        return

