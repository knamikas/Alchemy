"""The batch driver: everything between the CLI and the per-entry worker.

``main.py`` remains the documented entry point and holds the dispatch loop.
The modules here are the parts of it that have their own subject matter --
the output schemas and their writers, the progress line, and the run log.
"""
