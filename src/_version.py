"""Single source of the Alchemy version.

``pyproject.toml`` reads ``__version__`` from here via ``dynamic = ["version"]``
rather than declaring the number a second time, and ``main.py`` imports it for
the ``alchemy_version`` column recorded in every manifest row. Keeping one
definition means a released version and the provenance stamped into results
cannot disagree.

``importlib.metadata`` is deliberately not used: Alchemy is run from a clone and
is normally never installed, so distribution metadata is not available at
runtime.
"""

__version__ = "1.0.0"
