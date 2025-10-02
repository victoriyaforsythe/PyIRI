import numpy as np

import PyIRI
from PyIRI import sh_library


def test_output_shape():
    """Test that the output shape matches input shape."""
    solzen = np.random.uniform(0, 180, size=(10, 20, 2))
    prob = sh_library.Probability_F1_with_solzen(solzen)
    assert prob.shape == solzen.shape, "Output shape mismatch"


def test_range_of_output():
    """Test that the output is within expected probability range [0,1]."""
    solzen = np.linspace(0, 180, num=1000).reshape((10, 10, 10))
    prob = sh_library.Probability_F1_with_solzen(solzen)
    assert np.all((prob >= 0.0) & (prob <= 1.0)), "Probability out of range"


def test_known_values():
    """Test known input values and expected output."""
    # Zenith angle = 0° (cos(0)=1), so (1)^gamma = 1
    prob0 = sh_library.Probability_F1_with_solzen(np.array(0))
    assert np.isclose(prob0, 1.0), "Expected 1.0 at solzen = 0°"

    # Zenith angle = 90° (cos(90)=0), so (0.5)^gamma
    expected90 = (0.5)**2.36
    prob90 = sh_library.Probability_F1_with_solzen(np.array(90))
    assert np.isclose(prob90, expected90), f"Expected {expected90} at solzen = 90°"

    # Zenith angle = 180° (cos(180) = -1), so (0)^gamma = 0
    prob180 = sh_library.Probability_F1_with_solzen(np.array(180))
    assert np.isclose(prob180, 0.0), "Expected 0.0 at solzen = 180°"


def test_vectorized_behavior():
    """Test that function works on array input and scalar input."""
    solzen_scalar = 45
    solzen_array = np.array([45, 60, 90])
    prob_scalar = sh_library.Probability_F1_with_solzen(solzen_scalar)
    prob_array = sh_library.Probability_F1_with_solzen(solzen_array)

    assert np.isscalar(prob_scalar), "Scalar input did not return scalar"
    string = "Array input did not preserve shape"
    assert prob_array.shape == solzen_array.shape, string
