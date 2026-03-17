import os
import numpy as np


class FileObject:
    """A class to represent a file object with data and metadata."""

    def __init__(self, data=None, filename=None, dataDir=None, metadata=None, header=None, **kwargs):

        self.filename = filename
        self.dataDir = dataDir if dataDir else os.path.dirname(filename) if filename else None
        self.data = data if data is not None else np.array([])
        self.metadata = metadata if metadata is not None else {}
        self.header = header if header is not None else {}

        self.__dict__.update(kwargs)  # Allow additional attributes to be set dynamically

    def __repr__(self):
        return f'FileObject(filename={self.filename}, dataDir={self.dataDir})'


class DataObject:
    """A class to represent data which is manually added or generated. Used for internal or calibration uses."""

    def __init__(self, data=None, image_data=None, filename=None, dataDir=None, metadata=None, header=None, **kwargs):
        import numpy as _np

        self.data = data if data is not None else _np.array([])
        self.image_data = image_data if image_data is not None else _np.array([])
        self.filename = filename
        self.dataDir = dataDir if dataDir else os.path.dirname(filename) if filename else None
        self.metadata = metadata if metadata is not None else {}
        self.header = header if header is not None else {}

        self.__dict__.update(kwargs)  # Allow additional attributes to be set dynamically

    def __repr__(self):
        return f'DataObject(filename={self.filename}, dataDir={self.dataDir})'


__all__ = [
    'FileObject',
    'DataObject',
]
