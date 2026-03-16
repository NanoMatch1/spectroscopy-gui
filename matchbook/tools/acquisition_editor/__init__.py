"""Standalone acquisition editor for THz scan data.

Launch via::

    from matchbook.tools.acquisition_editor import edit_acquisitions
    edited = edit_acquisitions(raw_array)
"""

from matchbook.tools.acquisition_editor._editor import edit_acquisitions

__all__ = ["edit_acquisitions"]
