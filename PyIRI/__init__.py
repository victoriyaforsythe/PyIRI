"""Core library imports for PyIRI."""

import importlib

from PyIRI import igrf_library  # noqa F401
from PyIRI import main_library  # noqa F401
from PyIRI import plotting  # noqa F401

# Set version
__version__ = importlib.metadata.version('PyIRI')

# Determine the coefficient root directory
try:
    coeff_dir = str(importlib.resources.files(__package__).joinpath(
        'coefficients'))
except AttributeError:
    import os
    coeff_dir = os.path.join(os.path.realpath(os.path.dirname(__file__)),
                             'coefficients')
    del os

del importlib
