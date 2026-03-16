
import datetime
import json
from pathlib import Path
from typing import Any

import numpy as np


def _utc_now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _json_loads(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if not isinstance(value, str) or value.strip() == "":
        return {}
    try:
        loaded = json.loads(value)
    except Exception:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _safe_hdf5_key(name: str) -> str:
    return str(name).replace("/", "__")


class DatabaseService:
    """Generic HDF5 persistence utility for array-based datasets.

    This service intentionally does not depend on project-specific classes
    like `DataSet` or `THzData`. It only stores and loads numpy arrays plus
    dynamic metadata dictionaries.
    """

    SCHEMA_VERSION = "1.0"
    DEFAULT_REGISTRY_NAME = "thz_series_registry.json"

    def export_h5(
        self,
        path: str | Path,
        *,
        series_name: str | None,
        file_data_map: dict[str, np.ndarray],
        dataset_metadata: dict[str, Any] | None = None,
        per_file_metadata: dict[str, dict[str, Any]] | None = None,
        compression: str | None = "gzip",
        compression_opts: int = 1,
        store_acq_mask: bool = True,
        registry_path: str | Path | None = None,
        update_registry: bool = True,
    ) -> Path:
        """Write file arrays + metadata to one HDF5 container."""
        try:
            import h5py
        except ImportError as exc:
            raise ImportError("h5py is required for HDF5 export. Install with `pip install h5py`.") from exc

        output_path = Path(path).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        dataset_metadata = dataset_metadata or {}
        per_file_metadata = per_file_metadata or {}

        with h5py.File(output_path, "w") as h5_file:
            h5_file.attrs["schema_version"] = self.SCHEMA_VERSION
            h5_file.attrs["export_timestamp_utc"] = _utc_now_iso()
            h5_file.attrs["series_name"] = series_name or ""
            h5_file.attrs["dataset_metadata_json"] = _json_dumps(dataset_metadata)

            files_group = h5_file.create_group("files")

            for filename, raw_data in file_data_map.items():
                group_key = _safe_hdf5_key(filename)
                file_group = files_group.create_group(group_key)
                file_group.attrs["original_filename"] = filename

                raw_array = np.asarray(raw_data)

                create_kwargs: dict[str, Any] = {}
                if compression:
                    create_kwargs.update(
                        compression=compression,
                        compression_opts=compression_opts,
                        shuffle=True,
                        chunks=True,
                    )

                file_group.create_dataset("raw_data", data=raw_array, **create_kwargs)

                file_metadata = per_file_metadata.get(filename, {}) or {}
                file_group.attrs["file_metadata_json"] = _json_dumps(file_metadata)

                if store_acq_mask and raw_array.ndim == 2:
                    expected_acq_count = max(raw_array.shape[1] - 1, 0)
                    provided_mask = file_metadata.get("acq_mask", None)

                    if provided_mask is None:
                        acq_mask_array = np.ones((expected_acq_count,), dtype=bool)
                    else:
                        acq_mask_array = np.asarray(provided_mask, dtype=bool)
                        if acq_mask_array.shape != (expected_acq_count,):
                            file_group.attrs["acq_mask_shape_warning"] = (
                                f"mask shape {acq_mask_array.shape} != expected {(expected_acq_count,)}"
                            )

                    file_group.create_dataset("acq_mask", data=acq_mask_array)

        if update_registry and series_name:
            self.update_registry(
                series_name=series_name,
                export_path=str(output_path),
                registry_path=registry_path,
            )

        return output_path

    def load_h5(
        self,
        path: str | Path,
        *,
        strict_schema: bool = False,
    ) -> dict[str, Any]:
        """Load one HDF5 container into a plain Python payload."""
        try:
            import h5py
        except ImportError as exc:
            raise ImportError("h5py is required for HDF5 import. Install with `pip install h5py`.") from exc

        input_path = Path(path).expanduser().resolve()
        if not input_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {input_path}")

        payload: dict[str, Any] = {
            "schema_version": "",
            "series_name": "",
            "dataset_metadata": {},
            "files": {},
        }

        with h5py.File(input_path, "r") as h5_file:
            schema_version = str(h5_file.attrs.get("schema_version", ""))
            if strict_schema and schema_version != self.SCHEMA_VERSION:
                raise ValueError(f"Unsupported schema_version: {schema_version!r}")

            payload["schema_version"] = schema_version
            payload["series_name"] = str(h5_file.attrs.get("series_name", ""))
            payload["dataset_metadata"] = _json_loads(h5_file.attrs.get("dataset_metadata_json", ""))

            files_group = h5_file.get("files", None)
            if files_group is None:
                return payload

            for group_key in files_group.keys():
                file_group = files_group[group_key]
                original_filename = str(file_group.attrs.get("original_filename", group_key))

                file_payload: dict[str, Any] = {
                    "raw_data": np.array(file_group["raw_data"]),
                    "file_metadata": _json_loads(file_group.attrs.get("file_metadata_json", "")),
                    "acq_mask": np.array(file_group["acq_mask"], dtype=bool) if "acq_mask" in file_group else None,
                }

                payload["files"][original_filename] = file_payload

        return payload

    @classmethod
    def update_registry(
        cls,
        *,
        series_name: str,
        registry_path: str | Path | None = None,
        export_path: str | None = None,
        last_opened_path: str | None = None,
    ) -> Path:
        """Update JSON registry entry for a series."""
        registry_file = cls._resolve_registry_path(registry_path)

        if not series_name:
            return registry_file

        registry = cls._load_registry_file(registry_file)
        series_entry = registry.get(series_name, {})

        series_entry["series_name"] = series_name
        series_entry["updated_utc"] = _utc_now_iso()
        if export_path is not None:
            series_entry["last_export_path"] = export_path
        if last_opened_path is not None:
            series_entry["last_opened_path"] = last_opened_path

        registry[series_name] = series_entry
        registry_file.write_text(_json_dumps(registry), encoding="utf-8")
        return registry_file

    @classmethod
    def resolve_series_from_registry(
        cls,
        series_name: str,
        *,
        registry_path: str | Path | None = None,
        prefer: str = "last_export_path",
    ) -> Path | None:
        """Resolve a stored path for a series from registry."""
        registry_file = cls._resolve_registry_path(registry_path)
        if not registry_file.exists():
            return None

        registry = cls._load_registry_file(registry_file)
        series_entry = registry.get(series_name)
        if not isinstance(series_entry, dict):
            return None

        selected_path = (
            series_entry.get(prefer)
            or series_entry.get("last_export_path")
            or series_entry.get("last_opened_path")
        )
        return Path(selected_path).expanduser().resolve() if selected_path else None

    @classmethod
    def _resolve_registry_path(cls, registry_path: str | Path | None) -> Path:
        resolved = Path(registry_path).expanduser().resolve() if registry_path else (Path.cwd() / cls.DEFAULT_REGISTRY_NAME)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        return resolved

    @staticmethod
    def _load_registry_file(path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return loaded if isinstance(loaded, dict) else {}
