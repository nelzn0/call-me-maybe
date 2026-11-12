"""ANSI color codes and a helper that adds them to terminal output."""

import sys


RED = "31"
GREEN = "32"
YELLOW = "33"
BLUE = "34"
CYAN = "36"


def colorize(text: str, code: str) -> str:
    """Adds ANSI color codes, but only when printing to a terminal.

    Colors are also turned off when the NO_COLOR environment variable is set.

    Args:
        text: The text to color
        code: An ANSI color code, like RED or GREEN

    Returns:
        The text wrapped in color codes, or the plain text if colors are off
    """

    if sys.stdout.isatty():
        return f"\033[{code}m{text}\033[0m"
    return text
