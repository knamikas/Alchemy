"""Identify and assess metal-donor coordination.

Workers call ``analysis.run_bond_analysis``, which walks the stages in order:
``candidates`` and ``declared_connections`` discover contacts, ``eligibility``
and ``donor_chemistry`` decide which ones are first-sphere donors, ``geometry``,
``dpi`` and ``density_zscores`` score them, and ``site_environment`` and
``site_summary`` build the per-site context; ``contact_record`` holds the record
they pass along and ``schema`` defines the output columns. ``policy`` is the
single home for the radii and thresholds they share. Import the submodules
directly: this package exports nothing and imports nothing itself, so importing
one stage does not pull in the rest.
"""
