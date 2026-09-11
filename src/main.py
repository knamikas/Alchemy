#!/usr/bin/env python
"""Expose the CLI entry point for the Alchemy launcher."""

import sys

from cli import main

__all__ = ["main"]


if __name__ == "__main__":
    sys.exit(main())
