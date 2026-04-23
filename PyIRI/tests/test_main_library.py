"""Unit tests for PyIRI's main_library."""

import numpy as np

import PyIRI
import PyIRI.main_library as main
from PyIRI.main_library import to_numpy_array


class TestToNumpyArray:
    """Test TestToNumpyArray."""

    def test_scalar_input(self):
        """Test test_scalar_input 1-element."""
        result = to_numpy_array(5.5)
        assert isinstance(result, np.ndarray)
        assert result.shape == (1,)
        assert result[0] == 5.5

    def test_list_input(self):
        """Test test_scalar_input float."""
        result = to_numpy_array([1, 2, 3])
        np.testing.assert_array_equal(result, np.array([1.0, 2.0, 3.0]))
        assert result.dtype == float

    def test_array_input(self):
        """Test test_scalar_input int."""
        arr = np.array([1, 2, 3], dtype=int)
        result = to_numpy_array(arr)
        np.testing.assert_array_equal(result, arr.astype(float))

    def test_nested_list(self):
        """Test test_scalar_input 2d."""
        result = to_numpy_array([[1, 2], [3, 4]])
        assert result.shape == (2, 2)
        assert result.dtype == float


def test_fo_1day_interpolation():
    """Test linear interpolation of foF2."""
    F2, F1, E, Es, *_ = main.IRI_density_1day(2024,
                                              3,
                                              8,
                                              np.array([0]),
                                              np.array([20]),
                                              np.array([40]),
                                              np.array([0]),
                                              main.IG12_2_F107(40),
                                              PyIRI.coeff_dir,
                                              ccir_or_ursi=0)

    fo_1day = []
    true_fo = []
    for layer in [F2, F1, E, Es]:
        fo_1day.append(layer['fo'][0, 0])
        Nm_1day = layer['Nm'][0, 0]
        true_fo.append(main.den2freq(Nm_1day))

    assert fo_1day == true_fo, ("fo 1day interpolation error: fo = "
                                f"{fo_1day}, should be {true_fo}")


def test_solar_interpolate_ig12_midpoint():
    """IG12 interpolation at median F107 returns weighted average."""
    F_min = np.array([0., 100.])
    F_max = np.array([100., 200.])
    F107 = main.IG12_2_F107(50)
    result = main.solar_interpolate(F_min, F_max, F107, solidx='IG12')
    np.testing.assert_array_almost_equal(result, [50., 150.], decimal=4)


def test_solar_interpolate_r12_branch():
    """R12 branch executes and returns proper shape."""
    F_min = np.array([1., 2.])
    F_max = np.array([3., 4.])
    result = main.solar_interpolate(F_min, F_max, F107=100, solidx='R12')
    assert result.shape == F_min.shape
    assert np.all(np.isfinite(result))


def test_solar_interpolate_bounds():
    """Interpolation at solar min/max boundaries returns exact F_min/F_max."""
    F_min = np.array([10.])
    F_max = np.array([20.])
    low = main.solar_interpolate(F_min, F_max, F107=main.IG12_2_F107(0),
                                 solmin=0, solmax=100)
    high = main.solar_interpolate(F_min, F_max, F107=main.IG12_2_F107(100),
                                  solmin=0, solmax=100)
    np.testing.assert_almost_equal(low[0], 10.)
    np.testing.assert_almost_equal(high[0], 20.)


def test_solar_interpolation_of_dictionary_structure():
    """Dictionary interpolation preserves keys and handles axis swapping."""
    F = {
        'param1': np.random.rand(3, 4, 2),
        'param2': np.random.rand(3, 4, 2)
    }
    result = main.solar_interpolation_of_dictionary(F, F107=50)
    assert set(result.keys()) == {'param1', 'param2'}
    assert result['param1'].shape == (3, 4)


def test_solar_interpolation_of_dictionary_F2_shu2015_uses_ig12_for_hm():
    """SHU2015 hmF2 model triggers IG12 path for 'hm' key."""
    F = {
        'hm': np.random.rand(3, 4, 2),    # IG12 path (because SHU2015)
    }
    F_copy = F.copy()
    result = main.solar_interpolation_of_dictionary_F2(F, F107=50,
                                                       hmF2_model='SHU2015')
    truth = main.solar_interpolate(F_copy['hm'][:, :, 0], F_copy['hm'][:, :, 1],
                                   F107=50)
    assert all(k in result for k in F)
    assert result['hm'].shape == (3, 4)
    assert np.all(result['hm'] == truth)


def test_solar_interpolation_of_dictionary_F2_amtb2013_uses_r12_for_hm():
    """AMTB2013 hmF2 model triggers R12 path for 'hm' key."""
    F = {
        'hm': np.random.rand(3, 4, 2),    # R12 path (because AMTB2013)
    }
    F_copy = F.copy()
    result = main.solar_interpolation_of_dictionary_F2(F, F107=50,
                                                       hmF2_model='AMTB2013')
    truth = main.solar_interpolate(F_copy['hm'][:, :, 0], F_copy['hm'][:, :, 1],
                                   F107=50, solidx='R12')
    assert all(k in result for k in F)
    assert result['hm'].shape == (3, 4)
    assert np.all(result['hm'] == truth)


def test_freq2den_inverse_square():
    """Frequency conversion follows f^2 relationship."""
    assert main.freq2den(1.0) == 1.24e10
    assert main.freq2den(2.0) == 4 * 1.24e10
    np.testing.assert_array_equal(main.freq2den(np.array([1., 2.])),
                                  np.array([1.24e10, 4.96e10]))


def test_R12_F107_roundtrip():
    """R12 -> F107 -> R12 returns approximately original value."""
    original = np.array([0., 50., 100., 200.])
    f107 = main.R12_2_F107(original)
    recovered = main.F107_2_R12(f107)
    np.testing.assert_array_almost_equal(original, recovered, decimal=1)


def test_F107_2_R12_known_value():
    """Known conversion: R12=0 -> F107=63.75."""
    result = main.R12_2_F107(0)
    assert result == 63.75


def test_IG12_R12_v2_vs_v1():
    """Version 1 and 2 give different results."""
    r12 = 100
    ig12_v1 = main.R12_2_IG12(r12, v=1)
    ig12_v2 = main.R12_2_IG12(r12, v=2)
    assert ig12_v1 != ig12_v2


def test_IG12_R12_roundtrip_v2():
    """IG12 <-> R12 roundtrip for version 2."""
    original = np.array([10., 50., 100.])
    ig12 = main.R12_2_IG12(original, v=2)
    recovered = main.IG12_2_R12(ig12, v=2)
    np.testing.assert_array_almost_equal(original, recovered, decimal=1)


def test_F107_IG12_chain_consistency():
    """F107 -> R12 -> IG12 matches direct F107_2_IG12."""
    f107 = 100.
    r12 = main.F107_2_R12(f107)
    ig12_manual = main.R12_2_IG12(r12, v=2)
    ig12_direct = main.F107_2_IG12(f107, v=2)
    assert ig12_manual == ig12_direct


def test_IG12_2_F107_inverse():
    """IG12 <-> F107 inverse relationship."""
    original = 100.
    ig12 = main.F107_2_IG12(original, v=2)
    recovered = main.IG12_2_F107(ig12, v=2)
    np.testing.assert_approx_equal(original, recovered, significant=3)
