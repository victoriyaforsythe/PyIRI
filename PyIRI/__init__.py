"""Core library imports for PyIRI."""

from importlib import metadata
from importlib import resources
import logging

# Define a logger object to allow easier log handling
logging.raiseExceptions = False
logger = logging.getLogger('pyiri_logger')


# Import the package modules and top-level classes
from PyIRI import edp_update  # noqa F401
from PyIRI import igrf_library  # noqa F401
from PyIRI import main_library  # noqa F401
from PyIRI import plotting  # noqa F401

# Set version
__version__ = metadata.version('PyIRI')

# Determine the coefficient root directory
coeff_dir = str(resources.files(__package__).joinpath('coefficients'))
