"""Real-Lance read/write tests; only the ``test_spark_*`` tests require a Sail server."""

from __future__ import annotations

import inspect
import pickle
import random
from datetime import date
from decimal import Decimal
from unittest.mock import patch

import pyarrow as pa
import pytest
from pyspark.sql.types import (
    ArrayType,
    BooleanType,
    DateType,
    DecimalType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

lance = pytest.importorskip("lance")
ds = pytest.importorskip("pyspark.sql.datasource")
if not hasattr(ds, "EqualTo"):
    pytest.skip("Lance datasource requires PySpark 4.1+", allow_module_level=True)

from pysail.spark.datasource.lance import (
    LanceDataSource,
    _delete_staged_files,
    _filter_to_sql,
    _local_filesystem_path,
    _supports_null_struct_parents,
)

WRITE_SCHEMA = pa.schema([("id", pa.int64()), ("name", pa.string())])


@pytest.fixture
def dataset(tmp_path):
    table = pa.table(
        {
            "id": pa.array([1, 2, 3, 4, 5, 6], pa.int64()),
            "name": ["alice", "O'Brien", "a%_\\", None, "雪", "ALICE"],
            "value": pa.array([1, 2, None, 4, 5, None], pa.int64()),
            "score": [1.0, float("nan"), None, -float("inf"), float("inf"), 6.0],
            "nested": pa.array([{"x": i} for i in range(6)]),
            "payload": [b"x" * 8192] * 6,
        }
    )
    return lance.write_dataset(table, str(tmp_path / "test.lance"), max_rows_per_file=2, max_rows_per_group=2)


def make_reader(dataset, **options):
    source = LanceDataSource(options={"path": dataset.uri, **options})
    return source.reader(source.schema())


def read_batches(reader):
    return [batch for partition in reader.partitions() for batch in reader.read(partition)]


def make_writer(path, schema=WRITE_SCHEMA, overwrite=False, **options):
    source = LanceDataSource(options={"path": str(path), **options})
    return pickle.loads(pickle.dumps(source.writer(schema, overwrite)))


def write_rows(writer, rows):
    batches = [pa.RecordBatch.from_pylist(rows, schema=writer.schema)] if rows else []
    message = writer.write(iter(batches))
    # Sail pickles task outputs before the driver-side commit.
    return pickle.loads(pickle.dumps(message))


@pytest.mark.parametrize(
    ("predicate", "ids"),
    [
        (ds.EqualTo(("id",), 2), [2]),
        (ds.GreaterThan(("id",), 4), [5, 6]),
        (ds.GreaterThanOrEqual(("id",), 4), [4, 5, 6]),
        (ds.LessThan(("id",), 3), [1, 2]),
        (ds.LessThanOrEqual(("id",), 3), [1, 2, 3]),
        (ds.Not(ds.EqualTo(("value",), 2)), [1, 4, 5]),
        (ds.In(("value",), (1, 4, None)), [1, 4]),
        (ds.Not(ds.In(("value",), (1, None))), []),
        (ds.IsNull(("value",)), [3, 6]),
        (ds.IsNotNull(("value",)), [1, 2, 4, 5]),
        (ds.EqualNullSafe(("value",), None), [3, 6]),
        (ds.Not(ds.EqualNullSafe(("value",), 2)), [1, 3, 4, 5, 6]),
        (ds.EqualTo(("name",), "O'Brien"), [2]),
        (ds.StringStartsWith(("name",), "a"), [1, 3]),
        (ds.StringEndsWith(("name",), "ICE"), [6]),
        (ds.StringContains(("name",), "%_\\"), [3]),
        (ds.StringContains(("name",), "雪"), [5]),
        (ds.EqualTo(("name",), "' OR TRUE --"), []),
    ],
)
def test_filters_execute_in_lance(dataset, predicate, ids):
    reader = make_reader(dataset)
    assert list(reader.pushFilters([predicate])) == []
    reader.pruneColumns(pa.schema([dataset.schema.field("id")]))
    batches = read_batches(reader)
    assert [x for b in batches for x in b.column("id").to_pylist()] == ids
    assert all(b.schema.names == ["id"] for b in batches)


@pytest.mark.parametrize(
    "predicate",
    [
        ds.EqualTo(("score",), 1.0),
        ds.EqualTo(("score",), float("nan")),
        ds.GreaterThan(("score",), float("inf")),
        ds.Not(ds.EqualTo(("score",), 1.0)),
        ds.In(("id",), ()),
        ds.In(("id",), tuple(range(4097))),
        ds.In(("id",), (1, "2")),
        ds.EqualTo(("id",), 1 << 80),
        ds.EqualTo(("missing",), 1),
        ds.EqualTo(("id` OR TRUE",), 1),
        ds.EqualTo(("nested", "x"), 1),
        ds.EqualTo(("nested", "missing"), 1),
        ds.EqualTo(("id", "x"), 1),
    ],
)
def test_unsafe_filters_remain_residual(dataset, predicate):
    reader = make_reader(dataset)
    assert list(reader.pushFilters([predicate])) == [predicate]
    assert reader.filters == []
    assert sum(b.num_rows for b in read_batches(reader)) == dataset.count_rows()


def test_temporal_integer_encoding_not_guessed():
    schema = pa.schema([("date", pa.date32()), ("timestamp", pa.timestamp("us"))])
    assert _filter_to_sql(ds.EqualTo(("date",), 1), schema) is None
    assert _filter_to_sql(ds.EqualTo(("timestamp",), 1), schema) is None


def test_nested_null_parent_masks_child_values(tmp_path):
    nested = pa.StructArray.from_arrays([pa.array([1, 1, 2])], names=["x"], mask=pa.array([True, False, False]))
    table = pa.table({"id": [1, 2, 3], "nested": nested})
    dataset = lance.write_dataset(table, str(tmp_path / "nested.lance"))
    reader = make_reader(dataset)
    for predicate in [ds.EqualTo(("nested", "x"), 1), ds.Not(ds.EqualTo(("nested", "x"), 1))]:
        assert list(reader.pushFilters([predicate])) == [predicate]


def test_nonnullable_struct_path(tmp_path):
    schema = pa.schema([pa.field("nested", pa.struct([("x", pa.int64())]), nullable=False)])
    table = pa.Table.from_pydict({"nested": [{"x": 1}, {"x": 2}]}, schema=schema)
    dataset = lance.write_dataset(table, str(tmp_path / "struct.lance"))
    reader = make_reader(dataset)
    assert list(reader.pushFilters([ds.EqualTo(("nested", "x"), 1)])) == []
    assert pa.Table.from_batches(read_batches(reader)).to_pylist() == [{"nested": {"x": 1}}]


def test_boolean_and_unsigned_literals(tmp_path):
    table = pa.table({"id": pa.array([0, 2**64 - 1], pa.uint64()), "flag": pa.array([True, False])})
    dataset = lance.write_dataset(table, str(tmp_path / "typed.lance"))
    reader = make_reader(dataset)
    filters = [ds.EqualTo(("id",), 2**64 - 1), ds.EqualTo(("flag",), False)]
    # Older Lance SQL parsers reject large unsigned literals: leave those
    # residual while still pushing the independently supported Boolean filter.
    assert list(reader.pushFilters(filters)) in ([], [filters[0]])
    assert [x for b in read_batches(reader) for x in b.column("id").to_pylist()] == [2**64 - 1]


def test_projection_and_batch_size(dataset):
    reader = make_reader(dataset, batch_size="1", late_materialization='["payload"]')
    reader.pruneColumns(pa.schema([dataset.schema.field("name"), dataset.schema.field("id")]))
    batches = read_batches(reader)
    assert sum(b.num_rows for b in batches) == dataset.count_rows()
    assert all(b.num_rows <= 1 and b.schema.names == ["name", "id"] for b in batches)


def test_byte_batch_target_is_capability_checked(dataset):
    if "batch_size_bytes" not in inspect.signature(lance.LanceDataset.scanner).parameters:
        with pytest.raises(ValueError, match="does not support 'batch_size_bytes'"):
            make_reader(dataset, batch_size_bytes="16384")
    else:
        reader = make_reader(dataset, batch_size_bytes="16384")
        assert reader.options.scan["batch_size_bytes"] == 16384
        assert sum(b.num_rows for b in read_batches(reader)) == dataset.count_rows()


def test_empty_projection_preserves_cardinality(dataset):
    reader = make_reader(dataset)
    reader.pruneColumns(pa.schema([]))
    batches = read_batches(reader)
    assert sum(b.num_rows for b in batches) == dataset.count_rows()
    assert all(b.num_columns == 0 for b in batches)


def test_pinned_version_and_pickle(dataset):
    reader = make_reader(dataset, fragments_per_partition="2")
    partitions = reader.partitions()
    assert [len(p.fragment_ids) for p in partitions] == [2, 1]
    assert list(reader.pushFilters([ds.GreaterThan(("id",), 2)])) == []
    reader.pruneColumns(pa.schema([dataset.schema.field("id")]))
    # Planning pins the manifest; append must not leak into any worker scan.
    lance.write_dataset(dataset.to_table(), dataset.uri, mode="append")
    restored = pickle.loads(pickle.dumps(reader))
    parts = pickle.loads(pickle.dumps(partitions))
    ids = [x for p in parts for b in restored.read(p) for x in b.column("id").to_pylist()]
    assert ids == [3, 4, 5, 6]
    assert make_reader(dataset, version=str(dataset.version)).version == reader.version


def test_deletions_and_scalar_index(dataset):
    dataset.create_scalar_index("id", "BTREE")
    dataset.delete("id = 4")
    reader = make_reader(dataset, use_scalar_index="true", late_materialization="true")
    assert list(reader.pushFilters([ds.GreaterThan(("id",), 2)])) == []
    assert [x for b in read_batches(reader) for x in b.column("id").to_pylist()] == [3, 5, 6]


def test_filter_state_replaced_and_disabled(dataset):
    reader = make_reader(dataset)
    assert list(reader.pushFilters([ds.EqualTo(("id",), 1)])) == []
    assert list(reader.pushFilters([])) == []
    assert sum(b.num_rows for b in read_batches(reader)) == dataset.count_rows()
    disabled = make_reader(dataset, predicate_pushdown="false")
    predicate = ds.EqualTo(("id",), 1)
    assert list(disabled.pushFilters([predicate])) == [predicate]
    assert sum(b.num_rows for b in read_batches(disabled)) == dataset.count_rows()


def test_planner_rejection_keeps_filter(dataset):
    reader = make_reader(dataset)
    predicate = ds.EqualTo(("id",), 1)
    with patch.object(lance.LanceDataset, "scanner", side_effect=ValueError("unsupported syntax")):
        assert list(reader.pushFilters([predicate])) == [predicate]
    assert reader.filters == []


def test_empty_dataset_schema(tmp_path):
    dataset = lance.write_dataset(pa.table({"id": pa.array([], pa.int64())}), str(tmp_path / "empty.lance"))
    reader = make_reader(dataset)
    assert reader.schema == dataset.schema
    assert read_batches(reader) == []


def test_blob_values_or_explicit_unsupported_error(tmp_path):
    schema = pa.schema(
        [
            pa.field("id", pa.int64()),
            pa.field("blob", pa.large_binary(), metadata={b"lance-encoding:blob": b"true"}),
        ]
    )
    table = pa.Table.from_pydict({"id": [1, 2, 3], "blob": [b"abc", b"", None]}, schema=schema)
    dataset = lance.write_dataset(table, str(tmp_path / "blobs.lance"))
    reader = make_reader(dataset)
    for predicate in [ds.IsNull(("blob",)), ds.Not(ds.IsNull(("blob",))), ds.EqualNullSafe(("blob",), None)]:
        assert list(reader.pushFilters([predicate])) == [predicate]
    if "blob_handling" in inspect.signature(lance.LanceDataset.scanner).parameters:
        assert pa.Table.from_batches(read_batches(reader)).to_pydict() == table.to_pydict()
    else:
        with pytest.raises(ValueError, match="blob_handling"):
            read_batches(reader)
    # Even old pylance can prune unsupported blob columns without decoding.
    reader.pruneColumns(pa.schema([schema.field("id")]))
    assert pa.Table.from_batches(read_batches(reader)).to_pydict() == {"id": [1, 2, 3]}


def test_relative_path_survives_worker_cwd(dataset, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    source = LanceDataSource(options={"path": "test.lance"})
    reader = pickle.loads(pickle.dumps(source.reader(source.schema())))
    other = tmp_path / "worker"
    other.mkdir()
    monkeypatch.chdir(other)
    assert sum(b.num_rows for b in read_batches(reader)) == dataset.count_rows()


def test_scalar_index_includes_unindexed_append(dataset):
    dataset.create_scalar_index("id", "BTREE")
    lance.write_dataset(dataset.to_table().slice(0, 1), dataset.uri, mode="append")
    reader = make_reader(dataset)
    assert list(reader.pushFilters([ds.EqualTo(("id",), 1)])) == []
    assert [x for b in read_batches(reader) for x in b.column("id").to_pylist()] == [1, 1]


def test_projection_and_late_materialization_reduce_io(tmp_path):
    """Use native counters, not wall time: unrelated payloads must stay unread."""
    rng = random.Random(42)
    table = pa.table({"id": range(256), "payload": [rng.randbytes(65536) for _ in range(256)]})
    dataset = lance.write_dataset(table, str(tmp_path / "wide.lance"))

    def scan(columns, predicate):
        reader = make_reader(dataset, late_materialization="true")
        reader.pruneColumns(pa.schema([dataset.schema.field(c) for c in columns]))
        assert list(reader.pushFilters([predicate] if predicate else [])) == []
        stats = []
        reader.options.scan["scan_stats_callback"] = stats.append
        batches = read_batches(reader)
        return sum(b.num_rows for b in batches), sum(s.bytes_read for s in stats)

    full_rows, full_bytes = scan(["id", "payload"], None)
    projected_rows, projected_bytes = scan(["id"], None)
    filtered_rows, filtered_bytes = scan(["id", "payload"], ds.EqualTo(("id",), 42))
    assert full_rows == projected_rows == table.num_rows
    assert filtered_rows == 1
    assert 0 < projected_bytes < full_bytes / 2
    assert 0 < filtered_bytes < full_bytes / 2


def test_write_create_append_and_overwrite(tmp_path):
    uri = tmp_path / "write.lance"
    writer = make_writer(uri)
    assert isinstance(writer, ds.DataSourceArrowWriter)
    writer.commit([write_rows(writer, [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}])])
    dataset = lance.dataset(str(uri))
    assert dataset.version == 1
    assert dataset.to_table().to_pylist() == [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}]

    first_appender = make_writer(uri)
    second_appender = make_writer(uri)
    first_appender.commit(
        [
            write_rows(first_appender, [{"id": 3, "name": "c"}]),
            write_rows(second_appender, [{"id": 4, "name": "d"}]),
            None,
        ]
    )
    dataset = lance.dataset(str(uri))
    assert dataset.version == 2
    assert [row["id"] for row in dataset.to_table().to_pylist()] == [1, 2, 3, 4]

    overwriter = make_writer(uri, overwrite=True)
    overwriter.commit([write_rows(overwriter, [{"id": 5, "name": "e"}])])
    dataset = lance.dataset(str(uri))
    assert dataset.version == 3
    assert dataset.to_table().to_pylist() == [{"id": 5, "name": "e"}]


def test_write_empty_inputs(tmp_path):
    uri = tmp_path / "empty-write.lance"
    creator = make_writer(uri)
    creator.commit([creator.write(iter([]))])
    dataset = lance.dataset(str(uri))
    assert (dataset.version, dataset.count_rows()) == (1, 0)

    appender = make_writer(uri)
    appender.commit([appender.write(iter([]))])
    assert lance.dataset(str(uri)).version == 1

    overwriter = make_writer(uri, overwrite=True)
    overwriter.commit([overwriter.write(iter([]))])
    dataset = lance.dataset(str(uri))
    assert (dataset.version, dataset.count_rows()) == (2, 0)


def test_write_fragment_sizing_and_complex_types(tmp_path):
    uri = tmp_path / "sized.lance"
    schema = pa.schema(
        [
            pa.field("id", pa.int64()),
            pa.field("vector", pa.list_(pa.float32(), 2)),
            pa.field("blob", pa.large_binary(), metadata={b"lance-encoding:blob": b"true"}),
        ]
    )
    rows = [{"id": 1, "vector": [1.0, 2.0], "blob": b"abc"}, {"id": 2, "vector": None, "blob": None}]
    writer = make_writer(uri, schema=schema, max_rows_per_file="1", max_rows_per_group="1")
    message = write_rows(writer, rows)
    assert len(message) == 2
    writer.commit([message])
    reader = LanceDataSource(options={"path": str(uri)}).reader(lance.dataset(str(uri)).schema)
    if "blob_handling" in inspect.signature(lance.LanceDataset.scanner).parameters:
        assert pa.Table.from_batches(read_batches(reader)).to_pylist() == rows
    else:
        with pytest.raises(ValueError, match="blob_handling"):
            read_batches(reader)
    reader.pruneColumns(pa.schema([schema.field("id"), schema.field("vector")]))
    assert pa.Table.from_batches(read_batches(reader)).to_pylist() == [
        {"id": 1, "vector": [1.0, 2.0]},
        {"id": 2, "vector": None},
    ]


def test_write_null_struct_parent(tmp_path):
    uri = tmp_path / "null-struct.lance"
    schema = pa.schema([pa.field("point", pa.struct([("x", pa.float64())]), nullable=True)])
    if not _supports_null_struct_parents():
        with pytest.raises(ValueError, match="null struct parents"):
            make_writer(uri, schema=schema)
        return
    writer = make_writer(uri, schema=schema)
    writer.commit([write_rows(writer, [{"point": None}])])
    assert lance.dataset(str(uri)).to_table().to_pylist() == [{"point": None}]


def test_write_schema_mismatch(tmp_path):
    uri = tmp_path / "mismatch.lance"
    lance.write_dataset(pa.table({"id": [1]}), str(uri))
    with pytest.raises(ValueError, match="append schema"):
        make_writer(uri, schema=pa.schema([("id", pa.string())]))
    writer = make_writer(uri, schema=pa.schema([("id", pa.int64())]))
    bad = pa.RecordBatch.from_pylist([{"id": "x"}], schema=pa.schema([("id", pa.string())]))
    with pytest.raises(Exception, match="does not match the writer schema"):
        writer.write(iter([bad]))


def test_write_relative_path_survives_worker_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source = LanceDataSource(options={"path": "relative-write.lance"})
    writer = pickle.loads(pickle.dumps(source.writer(WRITE_SCHEMA, False)))
    worker = tmp_path / "worker"
    worker.mkdir()
    monkeypatch.chdir(worker)
    writer.commit([write_rows(writer, [{"id": 1, "name": "a"}])])
    assert (tmp_path / "relative-write.lance").exists()
    assert not (worker / "relative-write.lance").exists()


def test_write_abort_rejects_unsafe_paths(tmp_path):
    writer = make_writer(tmp_path / "unsafe.lance")
    with pytest.raises(ValueError, match="Invalid staged Lance file path"):
        writer.abort([[{"files": [{"path": "../outside.lance"}]}]])
    with pytest.raises(TypeError, match="Invalid Lance fragment commit message"):
        writer.abort([[{"files": "not-a-list"}]])


def test_delete_staged_files_supports_overlays_and_remote_versions(tmp_path, monkeypatch):
    uri = tmp_path / "cleanup.lance"
    staged = uri / "data" / "staged.lance"
    staged.parent.mkdir(parents=True)
    staged.write_bytes(b"staged")
    fragment = {"files": [], "overlays": [{"data_file": {"path": "staged.lance"}}]}
    _delete_staged_files(str(uri), [fragment], {})
    assert not staged.exists()
    assert _local_filesystem_path("C:\\data\\dataset.lance") == "C:\\data\\dataset.lance"
    monkeypatch.setattr("pysail.spark.datasource.lance._CAN_DELETE_STAGED_FILES", False)
    with pytest.raises(RuntimeError, match="upgrade pylance"):
        _delete_staged_files("s3://bucket/dataset.lance", [fragment], {})


def test_write_abort_removes_staged_files(tmp_path):
    uri = tmp_path / "aborted.lance"
    writer = make_writer(uri)
    message = write_rows(writer, [{"id": 1, "name": "a"}])
    staged = tmp_path / "aborted.lance" / "data" / message[0]["files"][0]["path"]
    assert staged.exists()
    writer.abort([message])
    assert not staged.exists()
    assert not (tmp_path / "aborted.lance" / "_versions").exists()


@pytest.mark.parametrize(
    "options",
    [
        {"version": "1"},
        {"batch_size": "1"},
        {"storage_options": "[]"},
        {"storage_options": "{"},
        {"max_rows_per_file": "0"},
        {"max_bytes_per_file": "not-an-integer"},
    ],
)
def test_invalid_write_options(options):
    with pytest.raises(ValueError):
        LanceDataSource(options={"path": "/unused", **options}).writer(WRITE_SCHEMA, False)


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"path": "/unused", "batch_size": "0"},
        {"path": "/unused", "batch_size": "1.2"},
        {"path": "/unused", "fragment_readahead": "-1"},
        {"path": "/unused", "fragments_per_partition": "0"},
        {"path": "/unused", "version": "-1"},
        {"path": "/unused", "use_scalar_index": "yes"},
        {"path": "/unused", "storage_options": "[]"},
        {"path": "/unused", "storage_options": '{"key": 1}'},
        {"path": "/unused", "late_materialization": "42"},
        {"path": "/unused", "late_materialization": "[1]"},
        {"path": "/unused", "max_rows_per_file": "1"},
        {"path": "/unused", "limit": "10"},
    ],
)
def test_invalid_options(options):
    with pytest.raises(ValueError):
        LanceDataSource(options=options).schema()


def test_spark_lance_query(spark, dataset):
    spark.dataSource.register(LanceDataSource)
    df = spark.read.format("lance").option("path", dataset.uri).load()
    assert df.count() == dataset.count_rows()
    # The float predicate is deliberately residual, and its column must survive
    # projection even though only "name" appears in the requested output.
    query = df.filter((df.id > 1) & (df.score < 5)).select("name")
    assert [row.name for row in query.collect()] == [None]
    assert [r.id for r in df.filter(df.name.contains("%_\\")).select("id").collect()] == [3]
    assert df.limit(2).count() == 2


def test_spark_write_roundtrip(spark, tmp_path):
    uri = str(tmp_path / "spark-write.lance")
    spark.dataSource.register(LanceDataSource)
    rows = [(i, f"name-{i}") for i in range(8)]
    spark.createDataFrame(rows, ["id", "name"]).repartition(4).write.format("lance").mode("append").save(uri)
    written = spark.read.format("lance").option("path", uri).load().orderBy("id").collect()
    assert [(row.id, row.name) for row in written] == rows
    assert lance.dataset(uri).version == 1

    spark.createDataFrame([(8, "name-8")], ["id", "name"]).write.format("lance").mode("append").save(uri)
    assert lance.dataset(uri).version == 2
    assert spark.read.format("lance").option("path", uri).load().count() == 9

    spark.createDataFrame([(1, 1.5)], ["id", "value"]).write.format("lance").mode("overwrite").save(uri)
    overwritten = spark.read.format("lance").option("path", uri).load().collect()
    assert [(row.id, row.value) for row in overwritten] == [(1, 1.5)]
    assert lance.dataset(uri).version == 3

    empty_uri = str(tmp_path / "spark-empty-write.lance")
    spark.createDataFrame([], "id INT, name STRING").write.format("lance").mode("append").save(empty_uri)
    empty = lance.dataset(empty_uri)
    assert (empty.version, empty.count_rows()) == (1, 0)


def test_spark_write_complex_types(spark, tmp_path):
    uri = str(tmp_path / "spark-types.lance")
    spark.dataSource.register(LanceDataSource)
    schema = StructType(
        [
            StructField("id", IntegerType(), True),
            StructField("flag", BooleanType(), True),
            StructField("amount", DecimalType(10, 2), True),
            StructField("day", DateType(), True),
            StructField("tags", ArrayType(StringType()), True),
            StructField("nested", StructType([StructField("x", IntegerType(), True)]), False),
        ]
    )
    rows = [
        (1, True, Decimal("12.34"), date(2024, 1, 2), ["a", "b"], (7,)),
        (None, None, None, None, None, (0,)),
    ]
    spark.createDataFrame(rows, schema).write.format("lance").mode("overwrite").save(uri)
    written = spark.read.format("lance").option("path", uri).load().orderBy("id").collect()
    assert [(row.id, row.flag, row.amount, row.day, row.tags, row.nested) for row in written] == [
        (None, None, None, None, None, (0,)),
        (1, True, Decimal("12.34"), date(2024, 1, 2), ["a", "b"], (7,)),
    ]


def test_spark_positional_load_path(spark, dataset):
    """`.load(path)` and `.option("path", path).load()` must read the same dataset."""
    spark.dataSource.register(LanceDataSource)
    positional = spark.read.format("lance").load(dataset.uri).orderBy("id").collect()
    optional = spark.read.format("lance").option("path", dataset.uri).load().orderBy("id").collect()
    assert [(row.id, row.name) for row in positional] == [(row.id, row.name) for row in optional]
    assert len(positional) == dataset.count_rows()


def test_spark_empty_lance(spark, tmp_path):
    dataset = lance.write_dataset(pa.table({"id": pa.array([], pa.int64())}), str(tmp_path / "empty.lance"))
    spark.dataSource.register(LanceDataSource)
    assert spark.read.format("lance").option("path", dataset.uri).load().count() == 0


def test_spark_projection_reaches_lance(spark, dataset, monkeypatch):
    """Require the new bridge, not merely correct results after Rust projection."""
    scans = []
    scanner = lance.LanceDataset.scanner

    def trace_scan(self, *args, **kwargs):
        scans.append(kwargs.get("columns"))
        return scanner(self, *args, **kwargs)

    monkeypatch.setattr(lance.LanceDataset, "scanner", trace_scan)
    spark.dataSource.register(LanceDataSource)
    df = spark.read.format("lance").option("path", dataset.uri).load()
    assert len(df.select("name").collect()) == dataset.count_rows()
    assert scans and all(columns == ["name"] for columns in scans)
    scans.clear()
    assert df.count() == dataset.count_rows()
    assert scans and all(columns == [] for columns in scans)


def test_spark_vectors_and_blobs(spark, tmp_path):
    if "blob_handling" not in inspect.signature(lance.LanceDataset.scanner).parameters:
        pytest.skip("Materialized blob scans require newer pylance")
    schema = pa.schema(
        [
            pa.field("id", pa.int64()),
            pa.field("vector", pa.list_(pa.float32(), 2)),
            pa.field("blob", pa.large_binary(), metadata={b"lance-encoding:blob": b"true"}),
        ]
    )
    table = pa.Table.from_pydict(
        {"id": [1, 2, 3], "vector": [[1, 2], [3, 4], None], "blob": [b"abc", b"", None]}, schema=schema
    )
    dataset = lance.write_dataset(table, str(tmp_path / "multimodal.lance"))
    spark.dataSource.register(LanceDataSource)
    df = spark.read.format("lance").option("path", dataset.uri).load()
    rows = df.orderBy("id").collect()
    assert [r.vector for r in rows] == [[1.0, 2.0], [3.0, 4.0], None]
    assert [r.blob for r in rows] == [b"abc", b"", None]
    assert [r.id for r in df.filter(df.blob.isNull()).select("id").collect()] == [3]
