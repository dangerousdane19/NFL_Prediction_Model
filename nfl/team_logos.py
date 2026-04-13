"""
NFL team logo paths.
Logos are stored as PNG files in assets/logos/{TEAM}.png.
"""
import os

_LOGOS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "logos")


def get_logo_path(team: str) -> "str | None":
    """Return the absolute path to a team's logo PNG, or None if not found."""
    path = os.path.join(_LOGOS_DIR, f"{team}.png")
    return path if os.path.exists(path) else None
