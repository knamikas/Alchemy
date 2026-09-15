"""The checkout root every default path is anchored to."""

import os

# src/paths.py -> src -> the checkout.
REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
