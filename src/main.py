#!/usr/bin/env python
"""Thin Python delegate for the repository-root ``./alchemy`` launcher."""

import sys

from cli import main


__all__ = ["main"]


if __name__ == "__main__":
    sys.exit(main())
