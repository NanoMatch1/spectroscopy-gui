import os
import numpy as np
import matplotlib.pyplot as plt

import dataset_core.io.loaders   # auto-imports all loader modules
from dataset_core.io import get_loader_for_extension
from dataset_core.data_structures.thz import THzData, BaseTHzData
from dataset_core.services.grouping import GroupingService
from dataset_core.services.database import DatabaseService
from pathlib import Path


class DataService:
    '''Custom dict-like object holds data and accesses it as needed. Holds master data dictionary and allows access to subsets via filename keys.'''
    
    def __init__(self, grouping_service: GroupingService | None = None):
        self._data_dict = {}
        self.grouping = grouping_service if grouping_service is not None else GroupingService()

    @property
    def references(self):
        return self.grouping.references

    @property
    def samples(self):
        return self.grouping.samples
    
    def is_reference(self, filename):
        return self.grouping.is_reference(filename)

    def is_sample(self, filename):
        return self.grouping.is_sample(filename)

    @property
    def data_dict(self) -> dict:
        '''Returns the full master data dictionary.'''
        return self._data_dict

    @property
    def info(self) -> str:
        '''Returns a summary string describing the DataService contents.'''
        lines = [f"DataService with {len(self._data_dict)} items."]
        for filename, item in self._data_dict.items():
            lines.append(str(item))
        return "\n".join(lines)

    def __repr__(self):
        return f"DataService with {len(self._data_dict)} items."

    def __getitem__(self, filename):
        return self._data_dict.get(filename, None)

    def __setitem__(self, filename, obj):
        self._data_dict[filename] = obj

    def __iter__(self):
        filelist = self.grouping.get_current_data_list()
        data_dict = {key: self._data_dict[key] for key in filelist}
        return iter(data_dict.values())
    
    def __len__(self):
        datadict = self.grouping.get_current_data_list()
        return len(datadict)
    
    def get(self, filename):
        return self._data_dict.get(filename, None)
    
    def keys(self):
        datadict = self.grouping.get_current_data_list()
        return datadict
    
    def values(self):
        filelist = self.grouping.get_current_data_list()
        data_dict = {key: self._data_dict[key] for key in filelist}
        return data_dict.values()

    def items(self):
        filelist = self.grouping.get_current_data_list()
        data_dict = {key: self._data_dict[key] for key in filelist}
        return data_dict.items()

    def add_items(self, mapping):
        self._data_dict.update(mapping)
    
    def add_item(self, filename, obj):
        self._data_dict[filename] = obj
    
    def remove_item(self, filename):
        if filename in self._data_dict:
            del self._data_dict[filename]

    def current_data_dict(self) -> dict:
        '''Returns data dict filtered to current filenames from the grouping service.'''
        current_filenames = self.grouping.get_current_data_list()
        return {key: self._data_dict[key] for key in current_filenames if key in self._data_dict}

    def update_filelist(self, filelist):
        self.grouping.update(filelist=filelist)

    def group_simple(self, **kwargs):
        '''Simple grouping based on sample and reference keys provided during initialization.'''
        result = self.grouping.simple_grouping(**kwargs)

        for filename, filename_info in self.grouping.file_items.items():
            data_obj = self._data_dict.get(filename, None)
            if data_obj is None:
                continue
            data_obj.filename_info = filename_info
        
        return result


class FigureObject:
    """Class for managing matplotlib figure and axis objects for plotting."""

    def __init__(self, figsize=None, title: str | None = None) -> None:
        self.figure, self.ax = plt.subplots(figsize=figsize)
        self.closed = False
        if title is not None:
            self.figure.suptitle(title)
        # track when this figure is closed
        self._cid = self.figure.canvas.mpl_connect('close_event', self._on_close)

    def _on_close(self, event):
        # mark as closed if this is our figure
        if event.canvas.figure is self.figure:
            self.closed = True

    def is_alive(self) -> bool:
        # double-check with matplotlib's figure registry
        return (not self.closed) and plt.fignum_exists(self.figure.number)

    def clear(self):
        self.ax.clear()

    def add_subplot(self, *args, **kwargs):
        ax = self.figure.add_subplot(*args, **kwargs)
        return ax


class Constants:

    """Class for defining physical constants and measurements used in THz data processing.

    Standard SI constants are loaded from scipy.constants.
    Pass experiment-specific values (thickness, ns, eps_inf, etc.) as keyword arguments.
    """

    def __init__(
        self,
        thickness: float | None = None,
        ns: float | None = None,
        eps_inf: float | None = None,
    ) -> None:
        import scipy.constants as cs

        self.speed_of_light = cs.c  # m/s
        self.hbar = cs.hbar  # J·s
        self.electron_charge = cs.e  # C
        self.permittivity_free_space = cs.epsilon_0  # F/m
        self.planck_constant = cs.h  # J·s
        Z0 = cs.physical_constants['characteristic impedance of vacuum'][0]

        self.impedance_free_space = Z0  # Ohm

        # Experiment-specific parameters
        self.thickness = thickness
        self.ns = ns
        self.eps_inf = eps_inf


class DataSet:
    '''Class for managing a dataset of THzData objects loaded from a directory.'''

    def __init__(
        self,
        file_dir: str,
        *,
        data_service: DataService | None = None,
        database_service: DatabaseService | None = None,
        sample_keys: list | None = None,
        reference_keys: list | None = None,
        seriesname: str | None = None,
        metadata: dict | None = None,
        file_metadata: dict | None = None,
    ) -> None:
        self.file_dir = file_dir
        self.data = data_service if data_service is not None else DataService()
        self.grouping = self.data.grouping  # alias for convenience
        self.sample_keys = sample_keys or []
        self.reference_keys = reference_keys or []
        self.figure_objects = {}
        self.seriesname = seriesname if seriesname is not None else Path(file_dir).name
        self.metadata = dict(metadata or {})
        self.file_metadata = dict(file_metadata or {})
        self.database_service = database_service if database_service is not None else DatabaseService()

        self.history = {}

    def set_dataset_metadata_tag(self, tag_name: str, tag_value):
        self.metadata[tag_name] = tag_value

    def remove_dataset_metadata_tag(self, tag_name: str):
        self.metadata.pop(tag_name, None)

    def set_file_metadata_tags(self, filename: str, **tags):
        current_tags = self.file_metadata.get(filename, {})
        current_tags.update(tags)
        self.file_metadata[filename] = current_tags

    def remove_file_metadata_tag(self, filename: str, tag_name: str):
        current_tags = self.file_metadata.get(filename, None)
        if not isinstance(current_tags, dict):
            return
        current_tags.pop(tag_name, None)

    def export_h5(
        self,
        path: str | Path,
        *,
        dataset_metadata: dict | None = None,
        per_file_metadata: dict[str, dict] | None = None,
        compression: str | None = 'gzip',
        compression_opts: int = 1,
        store_acq_mask: bool = True,
        registry_path: str | Path | None = None,
        update_registry: bool = True,
    ) -> Path:
        """Export current dataset to one HDF5 file."""
        file_data_map = {
            filename: np.asarray(data_object.raw_data)
            for filename, data_object in self.data.items()
        }

        merged_dataset_metadata = dict(self.metadata)
        merged_dataset_metadata.update(dataset_metadata or {})

        merged_per_file_metadata = {
            filename: dict(tags)
            for filename, tags in self.file_metadata.items()
            if isinstance(tags, dict)
        }
        for filename, tags in (per_file_metadata or {}).items():
            merged_per_file_metadata.setdefault(filename, {})
            merged_per_file_metadata[filename].update(tags or {})

        return self.database_service.export_h5(
            path=path,
            series_name=self.seriesname,
            file_data_map=file_data_map,
            dataset_metadata=merged_dataset_metadata,
            per_file_metadata=merged_per_file_metadata,
            compression=compression,
            compression_opts=compression_opts,
            store_acq_mask=store_acq_mask,
            registry_path=registry_path,
            update_registry=update_registry,
        )

    def import_h5(
        self,
        path: str | Path,
        *,
        strict_schema: bool = False,
        registry_path: str | Path | None = None,
        update_registry: bool = True,
    ):
        """Import dataset contents from one HDF5 file into this DataSet instance."""
        loaded_payload = self.database_service.load_h5(path=path, strict_schema=strict_schema)

        self.seriesname = loaded_payload.get('series_name', self.seriesname)
        self.metadata = dict(loaded_payload.get('dataset_metadata', {}) or {})
        self.file_metadata = {}

        self.data = DataService()
        self.grouping = self.data.grouping

        for filename, file_payload in loaded_payload.get('files', {}).items():
            raw_data = np.asarray(file_payload.get('raw_data'))
            data_object = self._build_thzdata_from_raw_array(raw_data=raw_data, filename=filename)
            self.data.add_item(filename, data_object)
            self.file_metadata[filename] = dict(file_payload.get('file_metadata', {}) or {})

        self.data.update_filelist(filelist=list(self.data.data_dict.keys()))

        if update_registry and self.seriesname:
            self.database_service.update_registry(
                series_name=self.seriesname,
                registry_path=registry_path,
                last_opened_path=str(Path(path).expanduser().resolve()),
            )

        return self

    def _build_thzdata_from_raw_array(self, raw_data: np.ndarray, filename: str) -> THzData:
        """Convert compiled raw array [time | acquisitions...] into THzData."""
        if raw_data.ndim != 2 or raw_data.shape[1] < 2:
            raise ValueError(
                f"Invalid raw_data shape for '{filename}': expected (N, >=2), got {raw_data.shape}."
            )

        scan_list = []
        time_axis = raw_data[:, 0]
        for acquisition_column_index in range(1, raw_data.shape[1]):
            scan_array = np.column_stack((time_axis, raw_data[:, acquisition_column_index]))
            scan_list.append(BaseTHzData(data=scan_array, headers=[]))

        return THzData(
            data=scan_list,
            header=[],
            filename=filename,
            data_type='h5',
        )

    def metadata_hook(self, filename: str, thzdata: THzData):
        """
        Hook you can implement later to pull rich metadata from your data service.

        Contract:
          - Return a JSON-serializable dict (or at least something json.dumps can handle with default=str).
          - Keep it "dynamic": arbitrary keys are fine.

        Example future implementations:
          - Look up filename in a SQLite DB / REST API
          - Pull instrument settings, operator, sample details, calibration refs, etc.
        """
        return {}

    @property
    def all_data(self) -> dict:
        """Returns the full master data dictionary (all loaded items)."""
        return self.data.data_dict

    @property
    def current_data(self) -> dict:
        """
        Returns a dict restricted to the filenames
        that the grouping service considers 'current'.
        """
        return self.data.current_data_dict()
    
    def group_simple(self, **kwargs):
        '''Simple grouping based on sample and reference keys provided during initialization.'''
        self.grouping.simple_grouping(sample_keys=self.sample_keys, reference_keys=self.reference_keys)
    
    def add_item(self, filename, obj):
        self.data.add_item(filename, obj)

    def remove_item(self, filename):
        self.data.remove_item(filename)

    def add_items(self, mapping):
        self.data.add_items(mapping)

    def set_current_files(self, filenames):
        self.grouping.set_current_data_list(list(filenames))

    def _generate_figure_object(self, key: str, **kwargs) -> FigureObject:
        """Get or create a FigureObject for a given key."""
        fig_obj = self.figure_objects.get(key, None)
        # If it doesn't exist or has been closed, make a new one
        if fig_obj is None or not fig_obj.is_alive():
            fig_obj = FigureObject(**kwargs)
            self.figure_objects[key] = fig_obj

        return fig_obj

    def load_any(self, path: str):
        ext = Path(path).suffix  # includes the dot
        Loader = get_loader_for_extension(ext)
        return Loader(path).load()

    def load_all_data(self, prefer_acc=True, file_lister=None) -> DataService:
        '''Loads all files in the specified directory into the data service.

        Parameters
        ----------
        prefer_acc : bool
            If True, acc files of the same name are loaded instead of dat files when both are present.
        file_lister : callable or None
            Optional function(directory_path) -> list[str] of filenames.
            Defaults to os.listdir. Inject a replacement for testing.
        '''
        if file_lister is None:
            file_lister = os.listdir

        raw_files = file_lister(self.file_dir)
        if prefer_acc:
            raw_files = [
                f for f in raw_files
                if not (f.endswith('.dat') and f[:-4] + '.acc' in raw_files)
            ]

        filelist = []
        for filename in raw_files:
            filepath = os.path.join(self.file_dir, filename)
            if os.path.isdir(filepath):
                continue  # skip folders
            data_object = self.load_any(filepath)
            self.data.add_item(filename, data_object)
            filelist.append(filename)

        if not filelist:
            raise FileNotFoundError(f"No valid data files found in {self.file_dir}. Please check the directory and file formats.")
        self.data.update_filelist(filelist=filelist)

        return self.data
    
    def load_constants(self, constants: Constants) -> None:
        '''Loads physical constants into all data objects in the dataset.'''
        self.constants = constants
        for data_object in self.data.values():
            data_object.constants = constants
        
    def grabonedata(self):
        '''Returns one data object from the dataset for quick access.'''
        if self.data:
            return next(iter(self.data.values()))
        print("Data dictionary is empty. Load data first.")
        return None

    def plot_current(self, **kwargs) -> None:
        '''Plots the current data for all data objects in the dataset.'''
        for name, data_object in self.data.items():
            figure_obj = self._generate_figure_object('main')
            data_object.plot_current(figure_obj=figure_obj, **kwargs)
        
        plt.show()

    def group_files(self, **kwargs):
        '''Groups files based on provided sample and reference keys.'''
        return self.data.group_simple(**kwargs)

    def assign_references(self, reference_type='substrate'):
        '''Workaround for convenience. Checks the filename_info from grouping and assigns the reference_filename attribute for each THzData object in the dataset.'''
        for filename, data_object in self.data.items():
            filename_info = data_object.filename_info
            if filename_info is not None:
                reference_filename = filename_info.substrate_reference
                if reference_filename is None and 'reference' in filename:
                    continue
                data_object.reference_filename = reference_filename

    def get_file_item(self, filename):
        '''Access the grouping information for a specific filename.'''
        return self.data.grouping(filename)
    
    def get_reference(self, filename, ref_type='substrate'):
        '''Consults the grouping service for the reference associated with the filename.
        ref_type returns either 'substrate' or 'air' reference.'''
        reference_filename = self.data.grouping.get_reference_filename(filename, ref_type=ref_type)
        if reference_filename is not None:
            return self.data.get(reference_filename)
        return None

    def modify_acquisitions(self, page_size: int = 25, in_place: bool = False, export=False, export_dir: str | Path | None = None) -> dict[str, np.ndarray]:
        from acquisition_editor import load_file, edit_data, save_acc, save_dat
        """Launch the standalone interactive acquisition editor.

        Requires that the loaded data objects expose a `raw_data` attribute
        (e.g. THzData objects).

        Constructs a dict of filename to raw_data arrays, passes it to the editor.

        Parameters
        ----------
        page_size : int
            Items shown per page in Included/Excluded lists.
        in_place : bool
            If True, applies saved edits back onto each object's raw_data.
            If False (default), leaves data objects untouched.

        Returns
        -------
        dict[str, np.ndarray]
            Mapping of filename to edited raw_data arrays.
        """
        def _construct_by_file_dict():
            raw_dict = {
                filename: np.asarray(data_object.raw_data)
                for filename, data_object in self.data.items()
            }
            
            return raw_dict

        edited_data = {}

        for filename, raw_data in _construct_by_file_dict().items():
            by_file_dict = {
                "filename": filename,
                "header": [],
                "scan_headers": [],
                "data": raw_data,
            }
            edited = edit_data(by_file_dict, page_size=page_size)
            edited_data[filename] = edited["data"]

        if in_place:
            for filename, new_raw in edited_data.items():
                data_object = self.data.get(filename)
                if data_object is not None:
                    print(f"Updating raw_data for '{filename}' with edited array of shape {new_raw.shape}.")
                    data_object.update_data(new_raw)
                else:
                    print(f"Warning: '{filename}' not found in data service. Skipping update.")

        if export:
            if export_dir is None:
                export_dir = os.path.join(self.file_dir, "export")
            for filename, new_raw in edited_data.items():
                save_data = {"filename": filename, "data": new_raw,}
                filepath = os.path.join(export_dir, filename)
                save_acc(save_data, filepath)
                save_dat(save_data, filepath)

        return edited_data
