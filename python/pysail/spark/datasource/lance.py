"""Streaming Lance reads and writes through ``pylance``, without a Rust Lance dependency.

Install ``pysail[lance]`` and PySpark 4.1+, then register ``LanceDataSource``::

    from pysail.spark.datasource.lance import LanceDataSource

    spark.dataSource.register(LanceDataSource)
    df = spark.read.format("lance").load("/data/events.lance")
    df.filter("tenant_id = 42").select("payload").show()

See ``lance.md`` alongside this module for options, pushdown guarantees, writes,
and the comparison with polars-pylance.
"""

from __future__ import annotations

import inspect
import json
import re
import tempfile
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING
from urllib.parse import urlparse
from urllib.request import url2pathname

import pyarrow as pa

try:
    import lance
    from lance.fragment import FragmentMetadata, write_fragments
except ImportError as e:
    msg = "pylance is required for the Lance data source. Install it with: pip install pysail[lance]"
    raise ImportError(msg) from e

try:
    from pyspark.sql import datasource as ds
    from pyspark.sql.datasource import EqualTo  # noqa: F401
except ImportError as e:
    msg = "The Lance data source requires PySpark 4.1+ with filter pushdown support."
    raise ImportError(msg) from e

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping
    from typing import Any


_COMPARISONS = {
    ds.EqualTo: "=",
    ds.GreaterThan: ">",
    ds.GreaterThanOrEqual: ">=",
    ds.LessThan: "<",
    ds.LessThanOrEqual: "<=",
}
_STRING_FUNCTIONS = {
    ds.StringStartsWith: "starts_with",
    ds.StringEndsWith: "ends_with",
    ds.StringContains: "contains",
}
_MAX_IN_VALUES = 4096
_SCANNER_PARAMETERS = inspect.signature(lance.LanceDataset.scanner).parameters
_FILE_SESSION = getattr(getattr(lance, "file", None), "LanceFileSession", None)
_CAN_DELETE_STAGED_FILES = _FILE_SESSION is not None and hasattr(_FILE_SESSION, "delete_file")
_SUPPORTS_NULL_STRUCT_PARENTS: bool | None = None
_SCAN_DEFAULTS = {
    "batch_size": 25_000,
    "batch_readahead": 1,
    "fragment_readahead": 1,
    "io_buffer_size": 32 * 1024 * 1024,
    "scan_in_order": True,
}


def _has_blob(field: pa.Field) -> bool:
    """Blob descriptors are not logical values, including for null predicates."""
    if (field.metadata or {}).get(b"lance-encoding:blob") == b"true":
        return True
    return any(_has_blob(field.type.field(i)) for i in range(field.type.num_fields))


def _column(attribute: object, schema: pa.Schema) -> tuple[str, pa.DataType] | None:
    """Resolve a column path without interpreting a literal name as SQL."""
    parts = (attribute,) if isinstance(attribute, str) else attribute
    if not isinstance(parts, tuple) or not parts:
        return None
    container = schema
    field = None
    for depth, part in enumerate(parts):
        # Lance does not reliably support embedded backticks, even when escaped.
        if not isinstance(part, str) or not part or "`" in part:
            return None
        if not isinstance(container, pa.Schema) and not pa.types.is_struct(container):
            return None
        index = container.get_field_index(part)
        if index < 0:
            return None
        field = container.field(index)
        if _has_blob(field):
            return None
        # Older Lance versions do not propagate a nullable parent's validity
        # into child predicates. Under NOT/IS NULL this can lose valid rows.
        if depth < len(parts) - 1 and field.nullable:
            return None
        container = field.type
    return ".".join(f"`{part}`" for part in parts), field.type


def _literal(value: object, dtype: pa.DataType) -> str | None:
    """Decline coercions whose semantics cannot be proven across the boundary.

    In particular, Sail currently serializes dates/timestamps as bare integers,
    and Spark's NaN ordering need not match the installed Lance/DataFusion.
    Neither temporal nor floating-point comparisons are pushed.
    """
    if value is None:
        return "NULL"
    if pa.types.is_boolean(dtype) and type(value) is bool:
        return "TRUE" if value else "FALSE"
    if pa.types.is_integer(dtype) and type(value) is int:
        bits = dtype.bit_width
        low = 0 if pa.types.is_unsigned_integer(dtype) else -(1 << (bits - 1))
        high = (1 << bits) - 1 if pa.types.is_unsigned_integer(dtype) else (1 << (bits - 1)) - 1
        if low <= value <= high:
            return str(value)
    if (pa.types.is_string(dtype) or pa.types.is_large_string(dtype)) and type(value) is str:
        return "'" + value.replace("'", "''") + "'"
    return None


def _filter_to_sql(f: ds.Filter, schema: pa.Schema) -> str | None:
    """Translate an exact supported filter, or leave it entirely to Sail.

    Never drop an unsupported child underneath NOT/OR. Independent filters
    passed to pushFilters are conjuncts and can be accepted separately.
    """
    if isinstance(f, ds.Not):
        child = _filter_to_sql(f.child, schema)
        return f"(NOT ({child}))" if child is not None else None

    # Current PySpark has no And/Or filter classes. Handle them if a future
    # version exposes them, without depending on private expression formats.
    for name in ("And", "Or"):
        cls = getattr(ds, name, None)
        if cls is not None and isinstance(f, cls):
            left = _filter_to_sql(f.left, schema)
            right = _filter_to_sql(f.right, schema)
            return f"({left} {name.upper()} {right})" if left is not None and right is not None else None

    column = _column(getattr(f, "attribute", None), schema)
    if column is None:
        return None
    name, dtype = column
    if isinstance(f, ds.IsNull):
        return f"({name} IS NULL)"
    if isinstance(f, ds.IsNotNull):
        return f"({name} IS NOT NULL)"
    if isinstance(f, ds.In):
        if not f.value or len(f.value) > _MAX_IN_VALUES:
            # Empty IN's null behavior is engine/version dependent under NOT.
            return None
        values = [_literal(v, dtype) for v in f.value]
        if any(v is None for v in values):
            return None
        return f"({name} IN ({', '.join(values)}))"
    op = _COMPARISONS.get(type(f))
    if op is not None or isinstance(f, ds.EqualNullSafe):
        value = _literal(f.value, dtype)
        if value is None:
            return None
        if isinstance(f, ds.EqualNullSafe):
            # Lance's SQL parser does not support IS NOT DISTINCT FROM.
            if f.value is None:
                return f"({name} IS NULL)"
            return f"({name} IS NOT NULL AND {name} = {value})"
        return f"({name} {op} {value})"
    function = _STRING_FUNCTIONS.get(type(f))
    if function is not None:
        value = _literal(f.value, dtype)
        if value is not None and isinstance(f.value, str):
            # Functions, not LIKE: %, _, and backslashes are literal characters.
            return f"{function}({name}, {value})"
    return None


def _normalize_dataset_path(path: str | None) -> str:
    """Require a dataset URI and make a local path independent of cwd."""
    if not path:
        msg = "Option 'path' is required for the lance data source"
        raise ValueError(msg)
    if not urlparse(path).scheme:
        return str(Path(path).resolve())
    return path


def _parse_storage_options(raw: str) -> dict[str, str]:
    try:
        options = json.loads(raw)
    except json.JSONDecodeError as e:
        msg = f"Lance option 'storage_options' must be valid JSON: {e}"
        raise ValueError(msg) from e
    if not isinstance(options, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in options.items()):
        msg = "Lance option 'storage_options' must be a JSON object of strings"
        raise ValueError(msg)
    return options


def _field_may_have_null_struct(field: pa.Field) -> bool:
    """Whether Arrow data can contain a null struct at this field's position."""
    dtype = field.type
    if pa.types.is_dictionary(dtype):
        return _field_may_have_null_struct(pa.field(field.name, dtype.value_type, field.nullable))
    if pa.types.is_struct(dtype):
        return field.nullable or any(_field_may_have_null_struct(child) for child in dtype)
    if pa.types.is_list(dtype) or pa.types.is_large_list(dtype) or pa.types.is_fixed_size_list(dtype):
        return _field_may_have_null_struct(dtype.value_field)
    return False


def _schema_may_have_null_struct(schema: pa.Schema) -> bool:
    return any(_field_may_have_null_struct(field) for field in schema)


def _supports_null_struct_parents() -> bool:
    """Probe whether this pylance preserves null struct parents on disk."""
    global _SUPPORTS_NULL_STRUCT_PARENTS
    if _SUPPORTS_NULL_STRUCT_PARENTS is None:
        try:
            with tempfile.TemporaryDirectory(prefix="lance-null-struct-probe") as directory:
                uri = str(Path(directory) / "probe.lance")
                table = pa.table({"point": pa.array([None], type=pa.struct([("x", pa.float64())]))})
                lance.write_dataset(table, uri)
                if lance.dataset(uri).to_table().to_pylist()[0]["point"] is None:
                    _SUPPORTS_NULL_STRUCT_PARENTS = True
                    return True
        except (OSError, ValueError, RuntimeError):
            pass
    return _SUPPORTS_NULL_STRUCT_PARENTS is True


def _local_filesystem_path(uri: str) -> str | None:
    """Return a local path for file URIs, or None for object-store URIs."""
    parsed = urlparse(uri)
    if not parsed.scheme or re.fullmatch(r"[A-Za-z]", parsed.scheme):
        return uri
    if parsed.scheme != "file" or parsed.netloc not in ("", "localhost"):
        return None
    return url2pathname(parsed.path)


def _boolean(name: str, value: str) -> bool:
    if value.lower() not in ("true", "false"):
        msg = f"Lance option {name!r} must be 'true' or 'false'"
        raise ValueError(msg)
    return value.lower() == "true"


def _integer(name: str, value: str, minimum: int = 1) -> int:
    try:
        result = int(value)
    except ValueError:
        result = minimum - 1
    if result < minimum:
        msg = f"Lance option {name!r} must be an integer >= {minimum}"
        raise ValueError(msg)
    return result


class _Options:
    """Validated, pickle-safe scan configuration (never an open Lance handle)."""

    def __init__(self, options: Mapping[str, str]) -> None:
        options = {k.lower(): v for k, v in options.items()}
        allowed = {
            "path",
            "version",
            "storage_options",
            "fragments_per_partition",
            "predicate_pushdown",
            "late_materialization",
            "use_scalar_index",
            "use_stats",
            "batch_size_bytes",
            *_SCAN_DEFAULTS,
        }
        unknown = options.keys() - allowed
        if unknown:
            msg = f"Unknown Lance options: {', '.join(sorted(unknown))}"
            raise ValueError(msg)
        self.path = _normalize_dataset_path(options.get("path"))
        self.version = _integer("version", options["version"]) if "version" in options else None
        self.storage_options = _parse_storage_options(options.get("storage_options", "{}"))
        self.fragments_per_partition = _integer("fragments_per_partition", options.get("fragments_per_partition", "1"))
        self.predicate_pushdown = _boolean("predicate_pushdown", options.get("predicate_pushdown", "true"))
        self.scan = dict(_SCAN_DEFAULTS)
        # New Lance versions otherwise return blob descriptors, not bytes.
        # Older versions are checked for output-schema compatibility in read().
        if "blob_handling" in _SCANNER_PARAMETERS:
            self.scan["blob_handling"] = "all_binary"
        if "batch_size_bytes" in options:
            if "batch_size_bytes" not in _SCANNER_PARAMETERS:
                msg = "Installed pylance does not support 'batch_size_bytes'; upgrade pylance or use 'batch_size'"
                raise ValueError(msg)
            self.scan["batch_size_bytes"] = _integer("batch_size_bytes", options["batch_size_bytes"])
        for name in ("batch_size", "io_buffer_size", "batch_readahead", "fragment_readahead"):
            if name in options:
                self.scan[name] = _integer(name, options[name], minimum=0 if name.endswith("readahead") else 1)
        for name in ("scan_in_order", "use_scalar_index", "use_stats"):
            if name in options:
                self.scan[name] = _boolean(name, options[name])
        if "late_materialization" in options:
            value = json.loads(options["late_materialization"])
            if not isinstance(value, bool) and not (isinstance(value, list) and all(isinstance(x, str) for x in value)):
                msg = "Lance option 'late_materialization' must be a JSON boolean or list of column names"
                raise ValueError(msg)
            self.scan["late_materialization"] = value

    def open(self, version: int | None = None):
        return lance.dataset(
            self.path,
            version=version if version is not None else self.version,
            storage_options=self.storage_options,
        )


class LancePartition(ds.InputPartition):
    """Fragment IDs at one immutable dataset version, not pickled native objects."""

    def __init__(self, version: int, fragment_ids: list[int]) -> None:
        super().__init__((version, fragment_ids))
        self.version = version
        self.fragment_ids = fragment_ids


class LanceReader(ds.DataSourceReader):
    """Projection-aware, streaming, fragment-parallel Lance reader."""

    def __init__(self, options: _Options, schema: pa.Schema) -> None:
        self.options = options
        dataset = options.open()
        if not dataset.schema.equals(schema, check_metadata=False):
            msg = "Lance schema changed or supplied schema differs; specify a dataset 'version' for a stable read"
            raise ValueError(msg)
        # Preserve Lance's resolved URI so a worker's cwd cannot change the
        # identity of a relative local dataset path.
        self.options.path = dataset.uri
        self.version = dataset.version
        self.schema = schema
        self.columns = schema.names
        self.filters: list[str] = []

    def pruneColumns(self, schema: pa.Schema) -> None:
        """Sail extension: emit only these columns, including residual inputs."""
        if any(self.schema.field(f.name).type != f.type for f in schema):
            msg = "Lance projection must preserve field types"
            raise ValueError(msg)
        self.columns = schema.names

    def pushFilters(self, filters: list[ds.Filter]) -> Iterator[ds.Filter]:
        accepted = []
        rejected = []
        dataset = self.options.open(self.version) if self.options.predicate_pushdown else None
        for f in filters:
            sql = _filter_to_sql(f, self.schema) if dataset is not None else None
            if sql is not None:
                try:
                    # Plan without pulling a batch. Decline syntax/functions not
                    # supported by this independently installed pylance version.
                    _ = dataset.scanner(columns=[], filter=sql).projected_schema
                except (ValueError, NotImplementedError):
                    sql = None
            if sql is None:
                rejected.append(f)
            else:
                accepted.append(sql)
        self.filters = accepted
        return iter(rejected)

    def partitions(self) -> list[LancePartition]:
        ids = [f.fragment_id for f in self.options.open(self.version).get_fragments()]
        size = self.options.fragments_per_partition
        return [LancePartition(self.version, ids[i : i + size]) for i in range(0, len(ids), size)]

    def read(self, partition: ds.InputPartition) -> Iterator[pa.RecordBatch]:
        if not isinstance(partition, LancePartition) or partition.version != self.version:
            msg = "Expected a LancePartition from this reader's dataset version"
            raise ValueError(msg)
        dataset = self.options.open(partition.version)
        fragments = [dataset.get_fragment(i) for i in partition.fragment_ids]
        if any(f is None for f in fragments):
            msg = "A planned Lance fragment is missing from the pinned dataset version"
            raise RuntimeError(msg)
        scanner = dataset.scanner(
            columns=self.columns,
            filter=" AND ".join(f"({f})" for f in self.filters) or None,
            fragments=fragments,
            **self.options.scan,
        )
        expected = pa.schema([self.schema.field(name) for name in self.columns])
        if not scanner.projected_schema.equals(expected, check_metadata=False):
            msg = (
                "Lance scanner output differs from the dataset schema. "
                "For blob columns, upgrade pylance to a version supporting blob_handling='all_binary', "
                "or select only non-blob columns."
            )
            raise ValueError(msg)
        # No to_table(), pandas, or Python rows. Preserve zero-column batch row
        # counts for COUNT(*) and release the native stream on cancellation.
        with scanner.to_reader() as batches:
            yield from batches


def _open_existing_dataset(path: str, storage_options: dict[str, str]):
    """Open a dataset, returning None when its manifest is absent."""
    try:
        return lance.dataset(path, storage_options=storage_options)
    except Exception as e:
        if isinstance(e, (ValueError, OSError)) and "was not found" in str(e):
            return None
        msg = f"Failed to open Lance dataset: {path!r}. Error: {e}"
        raise RuntimeError(msg) from e


def _staged_fragment_paths(fragment: Mapping[str, Any]) -> Iterator[str]:
    """Return dataset-relative paths for files staged by one write task."""
    files = fragment.get("files", [])
    overlays = fragment.get("overlays", [])
    if not isinstance(files, list) or not isinstance(overlays, list):
        msg = f"Invalid Lance fragment commit message: {fragment!r}"
        raise TypeError(msg)
    data_files = list(files)
    for overlay in overlays:
        if not isinstance(overlay, dict) or not isinstance(overlay.get("data_file"), dict):
            msg = f"Invalid Lance fragment commit message: {fragment!r}"
            raise TypeError(msg)
        data_files.append(overlay["data_file"])
    for data_file in data_files:
        if not isinstance(data_file, dict):
            msg = f"Invalid Lance fragment commit message: {fragment!r}"
            raise TypeError(msg)
        if data_file.get("base_id") is not None:
            msg = f"Cannot clean up a Lance fragment outside the dataset: {fragment!r}"
            raise ValueError(msg)
        path = data_file.get("path")
        relative = PurePosixPath(path) if isinstance(path, str) else PurePosixPath(".")
        if not isinstance(path, str) or not path or relative.is_absolute() or ".." in relative.parts:
            msg = f"Invalid staged Lance file path: {path!r}"
            raise ValueError(msg)
        yield (PurePosixPath("data") / path).as_posix()


def _delete_staged_files(uri: str, fragments: list[dict[str, Any]], storage_options: dict[str, str]) -> None:
    """Best-effort cleanup for fragments that were never committed."""
    paths = [path for fragment in fragments for path in _staged_fragment_paths(fragment)]
    if not paths:
        return
    local_path = _local_filesystem_path(uri)
    failures = []
    if local_path is not None:
        for path in paths:
            try:
                Path(local_path, *PurePosixPath(path).parts).unlink(missing_ok=True)
            except OSError as e:
                failures.append(f"{path}: {e}")
    else:
        if not _CAN_DELETE_STAGED_FILES:
            msg = "Installed pylance cannot delete uncommitted object-store files; upgrade pylance to clean up aborted writes"
            raise RuntimeError(msg)
        session = _FILE_SESSION(uri, storage_options=storage_options)
        for path in paths:
            try:
                session.delete_file(path)
            except OSError as e:
                if "not found" not in str(e).lower():
                    failures.append(f"{path}: {e}")
    if failures:
        msg = f"Failed to delete uncommitted Lance files: {'; '.join(failures)}"
        raise RuntimeError(msg)


class _WriteOptions:
    """Validated, pickle-safe write configuration (never an open Lance handle)."""

    def __init__(self, options: Mapping[str, str]) -> None:
        options = {k.lower(): v for k, v in options.items()}
        allowed = {
            "path",
            "storage_options",
            "max_rows_per_file",
            "max_rows_per_group",
            "max_bytes_per_file",
        }
        unknown = options.keys() - allowed
        if unknown:
            msg = f"Unknown Lance options: {', '.join(sorted(unknown))}"
            raise ValueError(msg)
        self.path = _normalize_dataset_path(options.get("path"))
        self.storage_options = _parse_storage_options(options.get("storage_options", "{}"))
        self.fragment_options = {
            "max_rows_per_file": _integer("max_rows_per_file", options.get("max_rows_per_file", "1048576")),
            "max_rows_per_group": _integer("max_rows_per_group", options.get("max_rows_per_group", "1024")),
            "max_bytes_per_file": _integer(
                "max_bytes_per_file", options.get("max_bytes_per_file", str(90 * 1024 * 1024 * 1024))
            ),
        }


class LanceArrowWriter(ds.DataSourceArrowWriter):
    """Streaming, fragment-parallel Lance writer with one dataset commit."""

    def __init__(self, options: _WriteOptions, schema: pa.Schema, overwrite: bool) -> None:
        self.options = options
        self.schema = schema
        self.overwrite = bool(overwrite)
        if _schema_may_have_null_struct(schema) and not _supports_null_struct_parents():
            msg = "Installed pylance does not preserve null struct parents; upgrade pylance to write nullable structs"
            raise ValueError(msg)
        existing = _open_existing_dataset(options.path, options.storage_options)
        if overwrite:
            self.fragment_mode = "overwrite"
            self.base_version = None
        elif existing is None:
            self.fragment_mode = "create"
            self.base_version = None
        else:
            if not existing.schema.equals(schema, check_metadata=False):
                msg = "Lance append schema does not match the existing dataset schema"
                raise ValueError(msg)
            self.options.path = existing.uri
            self.fragment_mode = "append"
            self.base_version = existing.version

    def _checked_batches(self, batches: Iterable[pa.RecordBatch]) -> Iterator[pa.RecordBatch]:
        for batch in batches:
            if not isinstance(batch, pa.RecordBatch):
                msg = f"Expected a PyArrow RecordBatch, got {type(batch)}"
                raise TypeError(msg)
            if not batch.schema.equals(self.schema):
                msg = "Lance write batch schema does not match the writer schema"
                raise ValueError(msg)
            yield batch

    def write(self, iterator: Iterable[pa.RecordBatch]) -> list[dict[str, Any]]:
        reader = pa.RecordBatchReader.from_batches(self.schema, self._checked_batches(iterator))
        fragments = write_fragments(
            reader,
            self.options.path,
            schema=self.schema,
            mode=self.fragment_mode,
            storage_options=self.options.storage_options,
            **self.options.fragment_options,
        )
        return [fragment.to_json() for fragment in fragments]

    def _fragment_dicts(self, messages: list[list[dict[str, Any]] | None] | None) -> list[dict[str, Any]]:
        if messages is None:
            return []
        fragments = []
        for message in messages:
            if message is None:
                continue
            if not isinstance(message, list) or not all(isinstance(item, dict) for item in message):
                msg = f"Invalid Lance fragment commit message: {message!r}"
                raise TypeError(msg)
            fragments.extend(message)
        return fragments

    def commit(self, messages: list[list[dict[str, Any]] | None]) -> None:
        fragments = [FragmentMetadata.from_json(json.dumps(item)) for item in self._fragment_dicts(messages)]
        if self.overwrite:
            operation = lance.LanceOperation.Overwrite(self.schema, fragments)
            lance.LanceDataset.commit(self.options.path, operation, storage_options=self.options.storage_options)
            return
        existing = _open_existing_dataset(self.options.path, self.options.storage_options)
        if existing is None:
            operation = lance.LanceOperation.Overwrite(self.schema, fragments)
            lance.LanceDataset.commit(self.options.path, operation, storage_options=self.options.storage_options)
            return
        if not existing.schema.equals(self.schema, check_metadata=False):
            msg = "Lance append schema does not match the existing dataset schema"
            raise ValueError(msg)
        if not fragments:
            return
        if self.base_version is not None and existing.uri == self.options.path:
            read_version = self.base_version
        else:
            read_version = existing.version
        operation = lance.LanceOperation.Append(fragments)
        lance.LanceDataset.commit(
            self.options.path,
            operation,
            read_version=read_version,
            storage_options=self.options.storage_options,
        )

    def abort(self, messages: list[list[dict[str, Any]] | None]) -> None:
        _delete_staged_files(self.options.path, self._fragment_dicts(messages), self.options.storage_options)


class LanceDataSource(ds.DataSource):
    """Lance datasets; see ``lance.md`` for the option reference."""

    @classmethod
    def name(cls) -> str:
        return "lance"

    def schema(self) -> pa.Schema:
        # Manifest-only schema discovery also works for empty datasets.
        return _Options(self.options).open().schema

    def reader(self, schema: pa.Schema) -> LanceReader:
        return LanceReader(_Options(self.options), schema)

    def writer(self, schema: pa.Schema, overwrite: bool) -> LanceArrowWriter:
        return LanceArrowWriter(_WriteOptions(self.options), schema, overwrite)
