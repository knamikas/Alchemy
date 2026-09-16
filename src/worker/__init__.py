"""The per-entry analysis that runs inside each pool process.

``lifecycle`` owns the process lifecycle and the entry result; ``resolve``,
``stages``, and ``memory`` are its parts; ``contracts`` holds the models that
cross the process boundary. Import the submodules directly: this package
imports nothing itself, so ``worker.contracts`` stays free of the analysis
stack for the driver modules that only read results.
"""
