"""Core library imports for PyIRI."""

# Define a logger object to allow easier log handling
import logging
logging.raiseExceptions = False
logger = logging.getLogger('pyiri_logger')

osflag = False
try:
    from importlib import metadata
    from importlib import resources
except ImportError:
    import importlib_metadata as metadata
    import os
    osflag = True

# Import the package modules and top-level classes
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
