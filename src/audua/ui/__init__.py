"""A small read-only web UI over an audua filetree.

``audua ui`` serves the tree at http://127.0.0.1:8765 and opens a browser: the
inbox and what is waiting in it, every run and its clips, and the summaries —
with clip audio playing and transcripts rendering in the page rather than
needing an external app.

The browser is the portable part. It is the one media player and markdown
viewer that is already installed on both Windows and Linux, so nothing here
shells out or depends on a desktop toolkit.
"""

from .server import make_server, serve
from .state import Roots, StateError

__all__ = ["Roots", "StateError", "make_server", "serve"]
