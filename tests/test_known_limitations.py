"""Validated-but-unfixed defects, pinned so a fix cannot land unnoticed.

Every executable defect pin in this module describes behaviour Alchemy *should*
have and is marked ``xfail(strict=True)``.  Unsafe process-level reproductions
are documented as explicit skips instead, so the reported outcome is:

``XFAIL``
    The defect is still present. This is the expected, quiet state.
``XPASS`` -> **the suite fails**
    Someone fixed the defect. That is good news, and it is deliberately loud:
    delete the ``xfail`` marker, move the test into the module that owns the
    behaviour, and update the finding list below.

Three properties keep this module honest:

* it asserts the *desired* behaviour, never the buggy behaviour, so a fix turns
  a red mark green rather than requiring the assertion to be inverted;
* nothing here needs CCP4 or the network -- CCP4 is stubbed with four
  do-nothing executables where the driver only probes ``PATH`` -- so the
  findings stay visible on a bare laptop checkout; every strict assertion is
  narrow enough that only a fix to *that* defect can satisfy it, because under
  ``strict=True`` an incidental pass is a red suite claiming a fix nobody made;
* findings that can only be reproduced by killing a process, filling a disk or
  exhausting memory are recorded as ``skip`` with the manual reproduction in the
  docstring, rather than as a fragile automated approximation.

One open finding remains, and it is pinned elsewhere: corrupt confidence
coverage, in ``test_confidence_score.py``.  This module currently owns no
pins -- every finding it held has been fixed and its test moved to the
module that owns the behaviour.  The protocol above is kept for the next
validated-but-unfixed defect.
"""

# No pins are currently open here. Every finding this module held has been
# fixed, and each regression test moved to the module that owns the behaviour:
#
#   --ccp4-setup precedence, --max-pdbs, --no-bonds help,
#   CCP4 config precedence          -> test_cli_and_config.py
#   SIGTERM cleanup, idle-worker
#   pool deadlock                   -> test_worker_recovery.py
#   leaked scratch directories,
#   unwritable --output-dir         -> test_driver_manifest.py
#   declared donor/NCS handling     -> test_declared_connections.py
#   placeholder unit cell           -> test_bond_geometry.py
#
# The one finding still open -- a corrupt geometry_coverage cell scoring as
# higher confidence -- is pinned in test_confidence_score.py, next to the
# scoring code it concerns.
#
# The protocol above stands for the next validated-but-unfixed defect: assert
# the desired behaviour, mark it xfail(strict=True) with a source anchor in the
# reason, and move it out when it turns green.
