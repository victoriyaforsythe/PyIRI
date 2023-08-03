"""Core library imports for PyIRI."""

import importlib

from PyIRI import igrf_library  # noqa F401
from PyIRI import main_library  # noqa F401
from PyIRI import pyiri  # noqa F401
from PyIRI import plotting  # noqa F401

# Set version
__version__ = importlib.metadata.version('PyIRI')

# Determine the coefficient root directory
coeff_dir = str(importlib.resources.files(__package__).joinpath('coefficients'))
