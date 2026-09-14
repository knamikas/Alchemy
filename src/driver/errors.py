"""The failure type the driver reports to the operator."""


class DriverError(Exception):
    """A user-facing driver failure: the message is reported and the run exits 1."""
