"""The batch driver: everything between the CLI and the per-entry worker.

``main.py`` remains the documented entry point as a shim over ``cli.py``.
``driver.pool`` owns batch orchestration and worker-pool supervision; the other
modules here own resume handling, output schemas and writers, resource probes,
the progress line, and the run log.
"""
