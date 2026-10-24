"""Keep all release-generated files inside this separate checkout."""
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]

def release_workspace():
    return REPOSITORY / '.runtime'

def release_upstream():
    return REPOSITORY / '.external' / 'mri_foundation'
