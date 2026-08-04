"""Metal-donor coordination: what contacts a metal has, and how good they are.

Six modules that answer one question, in the order a contact travels through
them. ``bond_analysis`` is the entry point -- ``run_bond_analysis`` -- and the
only one the worker calls.

``contact_record``       the record a candidate contact is, at every stage
``declared_connections`` source ``struct_conn``/``LINK`` claims, bound to the model
``donor_chemistry``      which residues and atoms may donate at all
``dpi``                  the precision index every z-score is measured against
``bond_schema``          the published CSV columns and the rows written to them

Nothing outside this package imports into it except ``worker``, which calls
``run_bond_analysis``, and the driver, which reads the schemas from
``bond_schema``. Nothing in here knows about density, entries or the pool.
"""
