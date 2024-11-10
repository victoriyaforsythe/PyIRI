# -*- coding: utf-8 -*-
"""Test the PyIRI IGRF Library functions.

Notes
-----
Functions that rely on the IGRF coefficients are expected have outputs that
change when the IGRF coefficients are updated.

"""

import numpy as np
import pytest

from PyIRI import coeff_dir
import PyIRI.igrf_library as ilib


class TestIGRFLibUtil(object):
    """Test class for igrf_library functions that do not use coefficients."""

    def setup_method(self):
        """Initialize for every test."""
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
                                 56.56854249, 70.71067812, 84.85281374,
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
        """Clean up after every test."""
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

    def test_bad_lengendre_poly_nmax(self):
        """Test Legendre polynomial expansion failure with bad index input."""

        with pytest.raises(IndexError) as ierr:
            ilib.legendre_poly(0, self.lat_in)

        assert str(ierr.value).find('out of bounds') >= 0, \
            "unexpected error message: {:s}".format(str(ierr.value))
        return

    @pytest.mark.parametrize('nmax', [1, 10, 100])
    def test_good_lengendre_poly(self, nmax):
        """Test Legendre polynomial expansion with different expansions.

        Parameters
        ----------
        nmax : int
            Positive integer specifying the degree of expansion.

        """
        # Get the polynomial expansion
        out = ilib.legendre_poly(nmax, self.lat_in)

        # Test the output shape
        assert out.shape == (nmax + 1, nmax + 2, 18)

        # Test select values
        assert np.all(out[0, 0] == 1.0)
        assert np.all(out[1, 1] == -out[0, 2])
        assert np.all(out[1, 2] == out[1, 0])
        assert np.isfinite(out).all()
        return


class TestIGRFLibCoeff(object):
    """Test class for igrf_library functions that use coefficients."""

    def setup_method(self):
        """Initialize for every test."""
        # Specify inputs
        self.date_in = 1981.1
        self.lon_in = np.arange(0.0, 360.0, 10.0)
        self.lat_in = np.zeros(shape=self.lon_in.shape)
        self.coeff_in = np.array([
            -2.996582e+04, -1.944780e+03, 5.581120e+03, -2.013500e+03,
            3.030740e+03, -2.143960e+03, 1.668280e+03, -2.233200e+02,
            1.284300e+03, -2.186160e+03, -3.302800e+02, 1.250120e+03,
            2.738600e+02, 8.321200e+02, -2.619000e+02, 9.375600e+02,
            7.815600e+02, 2.164000e+02, 3.898600e+02, -2.552400e+02,
            -4.201000e+02, 5.652000e+01, 1.926200e+02, -2.970000e+02,
            -2.171200e+02, 3.565600e+02, 4.622000e+01, 2.592400e+02,
            1.500000e+02, -7.818000e+01, -1.516600e+02, -1.624400e+02,
            -7.734000e+01, -4.756000e+01, 9.266000e+01, 4.910000e+01,
            6.578000e+01, -1.522000e+01, 4.398000e+01, 9.190000e+01,
            -1.904600e+02, 7.056000e+01, 4.000000e+00, -4.410000e+01,
            1.444000e+01, -1.780000e+00, -1.066800e+02, 1.788000e+01,
            7.244000e+01, -5.966000e+01, -8.222000e+01, 2.220000e+00,
            -2.700000e+01, 2.166000e+01, -4.340000e+00, -1.068000e+01,
            1.688000e+01, 1.660000e+00, 1.778000e+01, 1.078000e+01,
            -2.300000e+01, -1.560000e+00, -9.340000e+00, 1.866000e+01,
            6.000000e+00, 7.220000e+00, 0.000000e+00, -1.822000e+01,
            -1.100000e+01, 4.220000e+00, -7.440000e+00, -2.222000e+01,
            4.000000e+00, 9.440000e+00, 3.220000e+00, 1.556000e+01,
            5.560000e+00, -1.344000e+01, -1.660000e+00, -1.412000e+01,
            5.000000e+00, 1.000000e+01, -2.100000e+01, 1.000000e+00,
            1.578000e+01, -1.200000e+01, 9.000000e+00, 9.000000e+00,
            -5.220000e+00, -3.000000e+00, -6.000000e+00, -1.000000e+00,
            9.000000e+00, 7.000000e+00, 9.780000e+00, 1.780000e+00,
            -6.220000e+00, -5.000000e+00, 2.000000e+00, -4.000000e+00,
            -4.000000e+00, 1.000000e+00, 2.220000e+00, 0.000000e+00,
            -5.000000e+00, 3.000000e+00, -2.000000e+00, 6.000000e+00,
            5.000000e+00, -4.000000e+00, 3.000000e+00, 0.000000e+00,
            1.000000e+00, -1.000000e+00, 2.000000e+00, 4.000000e+00,
            3.000000e+00, 0.000000e+00, 0.000000e+00, -6.000000e+00,
            0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
            0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
            0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
            0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
            0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
            0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
            0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
            0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
            0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
            0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
            0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
            0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
            0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
            0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
            0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
            0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
            0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
            0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
            0.000000e+00, 0.000000e+00, 0.000000e+00])

        # Specify test outputs
        self.inc_out = np.array([-24.2087427, -25.34763124, -25.3937812,
                                 -24.17645495, -21.82609581, -19.57646011,
                                 -18.72214537, -19.31833925, -20.39518681,
                                 -20.8936541, -20.35910529, -19.2110154,
                                 -18.27920439, -17.86064041, -17.41913032,
                                 -16.21981989, -13.8294919, -10.16150189,
                                 -5.72201363, -1.61505912, 1.4037225,
                                 3.62906462, 5.68271573, 7.6671041,
                                 9.59477457, 11.9637194, 15.24712003,
                                 19.14098147, 22.501651, 23.73301264,
                                 21.40069455, 14.99625113, 5.35180003,
                                 -5.49742562, -14.95457288, -21.17926619])
        self.brad_out = -3.469099421742522e+17
        self.btheta_out = np.array([
            2.22713913e+17, 2.18112331e+17, 2.06883515e+17, 1.89368649e+17,
            1.66099912e+17, 1.37784313e+17, 1.05282208e+17, 6.95811562e+16,
            3.17659161e+16, -7.01451532e+15, -4.55818142e+16, -8.27641328e+16,
            -1.17431705e+17, -1.48531174e+17, -1.75117599e+17, -1.96383164e+17,
            -2.11681726e+17, -2.20548446e+17, -2.22713913e+17, -2.18112331e+17,
            -2.06883515e+17, -1.89368649e+17, -1.66099912e+17, -1.37784313e+17,
            -1.05282208e+17, -6.95811562e+16, -3.17659161e+16, 7.01451532e+15,
            4.55818142e+16, 8.27641328e+16, 1.17431705e+17, 1.48531174e+17,
            1.75117599e+17, 1.96383164e+17, 2.11681726e+17, 2.20548446e+17])
        self.bphi_out = np.array([
            -7.01451532e+15, -4.55818142e+16, -8.27641328e+16, -1.17431705e+17,
            -1.48531174e+17, -1.75117599e+17, -1.96383164e+17, -2.11681726e+17,
            -2.20548446e+17, -2.22713913e+17, -2.18112331e+17, -2.06883515e+17,
            -1.89368649e+17, -1.66099912e+17, -1.37784313e+17, -1.05282208e+17,
            -6.95811562e+16, -3.17659161e+16, 7.01451532e+15, 4.55818142e+16,
            8.27641328e+16, 1.17431705e+17, 1.48531174e+17, 1.75117599e+17,
            1.96383164e+17, 2.11681726e+17, 2.20548446e+17, 2.22713913e+17,
            2.18112331e+17, 2.06883515e+17, 1.89368649e+17, 1.66099912e+17,
            1.37784313e+17, 1.05282208e+17, 6.95811562e+16, 3.17659161e+16])

        self.test_out = None
        return

    def teardown_method(self):
        """Clean up after every test."""
        del self.date_in, self.lon_in, self.lat_in, self.inc_out, self.test_out
        del self.coeff_in
        return

    def test_inclination(self):
        """Test the inclination calculation."""
        self.test_out = ilib.inclination(coeff_dir, self.date_in, self.lon_in,
                                         self.lat_in)

        assert np.all(abs(self.test_out - self.inc_out) < 1.0e-4), \
            "Inclination differs by more than 4 significant figures"
        return

    def test_inclination_bad_coeff_dir(self):
        """Test the failure to calculate inclination with a bad IGRF path."""
        with pytest.raises(IOError) as ierr:
            ilib.inclination(".", self.date_in, self.lon_in, self.lat_in)

        assert str(ierr.value).find('unable to find IGRF coeff') >= 0, \
            "unexpected error message: {:s}".format(str(ierr.value))
        return

    def test_synth_values_defaults(self):
        """Test the calculated radial, co-lat, and azimuthal field values."""
        self.test_out = ilib.synth_values(self.coeff_in, 300.0, 0.0,
                                          self.lon_in)

        # Test the output shapes
        assert len(self.test_out) == 3
        assert self.test_out[0].shape == self.btheta_out.shape
        assert self.test_out[1].shape == self.btheta_out.shape
        assert self.test_out[2].shape == self.btheta_out.shape

        # Test the output values
        assert np.all(abs(self.test_out[0] - self.brad_out) < 1.0e-4), \
            "Radial outputs have an unexpected value"
        assert np.all(abs(self.test_out[1] == self.btheta_out) < 1.0e-4), \
            "Theta outputs have an unexpected value"
        assert np.all(abs(self.test_out[2] == self.bphi_out) < 1.0e-4), \
            "Phi outputs have an unexpected value"
        return

    def test_synth_values_grid(self):
        """Test the gridded radial, co-lat, and azimuthal field values."""
        self.test_out = ilib.synth_values(self.coeff_in, 300.0, 0.0,
                                          self.lon_in, grid=True)

        # Test the output shapes
        assert len(self.test_out) == 3
        assert self.test_out[0].shape == (1, self.btheta_out.shape[0])
        assert self.test_out[1].shape == (1, self.btheta_out.shape[0])
        assert self.test_out[2].shape == (1, self.btheta_out.shape[0])

        # Test the output values
        assert np.all(abs(self.test_out[0][0] - self.brad_out) < 1.0e-4), \
            "Radial outputs have an unexpected value"
        assert np.all(abs(self.test_out[1][0] == self.btheta_out) < 1.0e-4), \
            "Theta outputs have an unexpected value"
        assert np.all(abs(self.test_out[2][0] == self.bphi_out) < 1.0e-4), \
            "Phi outputs have an unexpected value"
        return

    def test_synth_values_bad_colat(self):
        """Test the calculated field values with bad co-latitude."""
        with pytest.raises(ValueError) as verr:
            ilib.synth_values(self.coeff_in, 300.0, -1.0, self.lon_in)

        assert str(verr.value).find('Colatitude outside bounds') >= 0, \
            "unexpected error message: {:s}".format(str(verr.value))
        return

    def test_synth_values_bad_expansion_min(self):
        """Test the calculated field values with bad nmin."""
        with pytest.raises(ValueError) as verr:
            ilib.synth_values(self.coeff_in, 300.0, 0.0, self.lon_in, nmin=-1)

        assert str(verr.value).find('Only positive nmin allowed') >= 0, \
            "unexpected error message: {:s}".format(str(verr.value))
        return

    def test_synth_values_bad_expansion_max(self):
        """Test the calculated field values with bad nmax."""
        with pytest.raises(ValueError) as verr:
            ilib.synth_values(self.coeff_in, 300.0, 0.0, self.lon_in, nmax=0)

        assert str(verr.value).find('Only positive, non-zero nmax') >= 0, \
            "unexpected error message: {:s}".format(str(verr.value))
        return

    def test_synth_values_bad_expansion_range(self):
        """Test the calculated field values with bad nmin-nmax combination."""
        with pytest.raises(ValueError) as verr:
            ilib.synth_values(self.coeff_in, 300.0, 0.0, self.lon_in, nmax=10,
                              nmin=20)

        assert str(verr.value).find('Nothing to compute') >= 0, \
            "unexpected error message: {:s}".format(str(verr.value))
        return

    def test_synth_values_bad_input_shape(self):
        """Test the calculated field values with bad input shapes."""
        with pytest.raises(ValueError) as verr:
            ilib.synth_values(self.coeff_in, 300.0, [0.0, 1.0], self.lon_in)

        assert str(verr.value).find('Cannot broadcast grid shapes') >= 0, \
            "unexpected error message: {:s}".format(str(verr.value))
        return
