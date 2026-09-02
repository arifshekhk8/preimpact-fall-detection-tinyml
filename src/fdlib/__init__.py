"""fdlib -- shared library for the pre-impact fall detection project.

Imported identically by the Kaggle notebooks and by the local firmware generator, so
that preprocessing can never drift between training and deployment. Mirrored to the
Kaggle Dataset `arifshekh/fdlib` by `scripts/sync_fdlib.py`; every notebook records
the version it ran against.
"""

__version__ = "0.1.0"

from . import config  # noqa: F401
