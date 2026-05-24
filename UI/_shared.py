"""
CV Builder AI — UI/_shared.py
Re-exports the helpers that all three UI PDF builders depend on.
These functions are defined in main.py; this thin shim lets the UI modules
import them with a clean path (from UI._shared import ...) without circular
imports, because main.py imports *from* the UI package, never the other way.
"""

# These names are injected by main.py at startup via UI._shared.<name> = <fn>.
# The assignments below are placeholders that satisfy static analysers and
# provide a clear error message if a UI module is accidentally imported before
# main.py has run the injection step.

def _normalise_edu_entry(e):
    raise RuntimeError("UI._shared not initialised — import main before using UI builders.")

def _infer_degree_duration(degree_str):
    raise RuntimeError("UI._shared not initialised — import main before using UI builders.")

def _contact_href(val):
    raise RuntimeError("UI._shared not initialised — import main before using UI builders.")
