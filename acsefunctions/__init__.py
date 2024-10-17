from acsefunctions import *

try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:  # For Python < 3.8
    from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    # package is not installed
    pass
