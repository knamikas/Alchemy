"""Single source of the Alchemy version.

``importlib.metadata`` cannot be used: Alchemy is run from a clone and is
normally never installed, so distribution metadata is absent at runtime.
"""

__version__ = "1.0.0"
