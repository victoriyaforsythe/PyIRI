"""Core library imports for PyIRI."""

try:
    import importlib.metadata as metadata
except ImportError:
    import importlib_metadata as metadata

try:
    import importlib.resources as resources
except ImportError:
    import os
    resources = None

from PyIRI import igrf_library  # noqa F401
from PyIRI import main_library  # noqa F401
from PyIRI import plotting  # noqa F401

# Set version
__version__ = metadata.version('PyIRI')

# Determine the coefficient root directory
if resources is None:
    coeff_dir = os.path.join(os.path.realpath(os.path.dirname(__file__)),
                             'coefficients')
    del os
else:
    coeff_dir = str(importlib.resources.files(__package__).joinpath(
        'coefficients'))

del metadata, resources
