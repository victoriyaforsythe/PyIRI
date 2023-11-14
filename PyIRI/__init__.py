"""Core library imports for PyIRI."""

osflag = False
try:
    from importlib import metadata
    from importlib import resources
except ImportError:
    import importlib_metadata as metadata
    import os
    osflag = True

from PyIRI import igrf_library  # noqa F401
from PyIRI import main_library  # noqa F401
from PyIRI import plotting  # noqa F401

# Set version
__version__ = metadata.version('PyIRI')

# Determine the coefficient root directory
if osflag:
    coeff_dir = os.path.join(os.path.realpath(os.path.dirname(__file__)),
                             'coefficients')
else:
    coeff_dir = str(resources.files(__package__).joinpath(
        'coefficients'))

del osflag
