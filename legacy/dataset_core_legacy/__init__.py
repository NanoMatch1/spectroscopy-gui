"""dataset_core — reusable data-container and I/O toolkit.

Extract this package out of the parent project to reuse loaders,
data containers, grouping, and persistence without any
analysis-specific processing code.
"""

from dataset_core.dataset import DataSet, DataService, FigureObject, Constants
from dataset_core.data_structures.thz import THzData, BaseTHzData
from dataset_core.data_structures.spectrum import Spectrum
from dataset_core.data_structures.common import FileObject, DataObject
from dataset_core.data_structures.filename_info import FilenameInfo
from dataset_core.io import get_loader_for_extension
from dataset_core.services.grouping import GroupingService
from dataset_core.services.database import DatabaseService
from dataset_core.services.provenance import record_provenance

__all__ = [
    "DataSet",
    "DataService",
    "FigureObject",
    "Constants",
    "THzData",
    "BaseTHzData",
    "Spectrum",
    "FileObject",
    "DataObject",
    "FilenameInfo",
    "get_loader_for_extension",
    "GroupingService",
    "DatabaseService",
    "record_provenance",
]
