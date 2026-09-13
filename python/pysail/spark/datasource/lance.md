# Lance Python datasource

This plugin calls the independently installed **`pylance`** wheel for reads and writes.
It adds no Lance dependency to `Cargo.toml` or `Cargo.lock`. Sail and Lance
exchange batches through PyArrow and Sail's existing Arrow C Data Interface,
not through a shared Rust `arrow-rs` or DataFusion dependency graph.

This removes the **Rust build-time version coupling**, not every compatibility
constraint: the Python runtime, PyArrow's types, the datasource API, and each
wheel's platform support must still be compatible. Install the plugin and
`pylance` in the Sail server and every worker's Python environment, not just
the Spark Connect client. Local paths must be accessible from every worker.

## Usage

```sh
pip install 'pysail[lance]' 'pyspark-client>=4.1'
```

```python
from pysail.spark.datasource.lance import LanceDataSource

spark.dataSource.register(LanceDataSource)

events = (
    spark.read.format("lance")
    .option("late_materialization", '["payload", "embedding"]')
    .option("batch_size", "8192")
    .option("path", "/data/events.lance")
    .load()
)
filtered = events.filter("tenant_id = 42").select("payload")
filtered.show()

# Each task stages fragments; Sail publishes them with one dataset commit.
filtered.write.format("lance").mode("append").save("/data/filtered.lance")
```

Use only existing column names in `late_materialization`; omit it to use
Lance's heuristic. SQL works through a registered temporary view as usual.
Registration is explicit, like the Vortex datasource; it does not override a
native Lance provider registered by another build.

The plugin works without the projection bridge change on older Sail versions
that support Python datasources, but those versions still read every output
column. Automatic column pruning requires the optional `pruneColumns` hook
included in this branch. This is a **Sail extension**, not a PySpark 4.1/4.2 API.
The plugin is intended for Sail, not the JVM Spark Python datasource runtime.
Both `.load("/data/events.lance")` and `.option("path", ...).load()` work:
Sail's Python datasource adapter forwards a single positional load path as
the `path` option, matching PySpark (an explicit `path` option takes
precedence if both are given). Positional `.save(path)` is likewise forwarded
for writes.

For writes, request one of the two supported Spark save modes explicitly:

```python
filtered.write.format("lance").mode("append").save("/data/filtered.lance")
```

## Options

Names are case-insensitive; values are Spark datasource strings. Unknown
options are rejected, so a misspelled memory-control option cannot silently
leave an expensive default enabled.

| Option | Default | Meaning |
| --- | --- | --- |
| `path` | Required | Lance **dataset directory/URI**, not a single fragment file |
| `version` | Latest at reader planning | Read-only positive integer dataset version |
| `storage_options` | `{}` | JSON object of string options forwarded to Lance |
| `fragments_per_partition` | `1` | Read-only number of fragments assigned to each Sail task |
| `predicate_pushdown` | `true` | Read-only diagnostic switch for predicate pushdown |
| `batch_size` | `25000` | Read-only maximum rows per Arrow batch |
| `batch_size_bytes` | Unset | Read-only decoded batch byte target; requires a supporting pylance (e.g. 11.0) |
| `batch_readahead` | `1` | Read-only Lance batch decode read-ahead |
| `fragment_readahead` | `1` | Read-only Lance fragment read-ahead within each task |
| `io_buffer_size` | `33554432` | Read-only per-scanner I/O buffer bytes (32 MiB) |
| `scan_in_order` | `true` | Read-only ordered batches within a task; **not** global SQL ordering |
| `late_materialization` | Lance heuristic | Read-only JSON `true`, `false`, or list of column names |
| `use_scalar_index` | Lance default | Read-only scalar-index switch |
| `use_stats` | Lance default | Read-only statistics-pruning switch |
| `max_rows_per_file` | `1048576` | Write-only maximum rows per staged data file |
| `max_rows_per_group` | `1024` | Write-only maximum rows per Lance group |
| `max_bytes_per_file` | `96636764160` | Write-only soft byte target per staged file |

Read planning rejects write-only tuning options, and write planning rejects
read-only options such as `version` and scanner tuning. This prevents a
performance- or correctness-sensitive setting from being silently ignored.

Do not store credentials in source files or committed examples. Prefer the
worker environment's cloud credential provider. Explicit `storage_options`
are serialized with readers and writers and must be treated as sensitive.

The bounded read-ahead defaults follow polars-pylance's memory-conscious
configuration. These are **not a total query memory limit**: concurrent Sail
tasks multiply scanner buffers, and decoded Arrow batches and result buffers
consume additional memory. Increase `fragments_per_partition` to reduce task
overhead for many small fragments; one large fragment cannot currently be
split across Sail tasks. SQL without `ORDER BY` has no global row order.
For large blobs, use much smaller row batches and, on supported pylance
versions, `batch_size_bytes`. A single oversized value may exceed the byte
target. Unsupported byte-target requests fail rather than being ignored.

## Writes

Supported explicit modes are:

- `.mode("append")`: append when the destination exists, otherwise create it.
- `.mode("overwrite")`: replace the destination when it exists, otherwise create it.

Sail's Python writer protocol currently passes only an `overwrite` Boolean to
Python. Consequently, non-overwrite Spark modes such as default, `error`, and
`ignore` currently reach this writer exactly like `append`. Do not rely on
them for existence checks. Likewise, `partitionBy` is not implemented for
Lance: partition columns would be stored as ordinary columns.

Each successful Sail task streams its Arrow batches into one or more
uncommitted Lance fragments and returns only fragment metadata. The driver
then publishes all successful fragments with a single Lance `Append` or
`Overwrite` transaction. Tasks never call `write_dataset()` independently.
Appends require the same ordered field names, logical types, and nullability
as the destination; Arrow metadata is ignored for that compatibility check.
Empty appends do not create a new dataset version; empty creates and
overwrites establish the requested schema. Writes do not create scalar
indexes. Concurrent writers use Lance's normal commit-conflict handling; no
separate cross-task lock is added, so concurrent append order is
nondeterministic. Schemas that may contain a null struct parent are checked
against the installed `pylance`: versions that do not preserve that null
fail during writer planning instead of storing default child values.

Aborts delete the exact uncommitted data files reported by successful tasks.
A task that fails before returning its fragment metadata can leave orphaned
staged files, as can a worker/process crash. The installed Lance version also
matters for non-local aborted writes: modern `pylance` can delete staged
object-store files, while older versions may leave them behind.

### Blob compatibility

Modern pylance is explicitly configured with `blob_handling="all_binary"`:
blob columns must yield their actual bytes, including distinct empty and null
values, not internal position/size descriptors. Old pylance (including 0.39)
cannot stream materialized blobs through this API. Selecting such a column
fails with an upgrade message; projecting it away still works. Output schema
compatibility is checked before any batch is yielded. Blob predicates remain
residual, because null descriptor semantics are not equivalent to logical
binary null semantics.

## What avoids decoding

1. **Manifest-only schema discovery.** Empty datasets work without reading a
   first data batch.
2. **Projection before decoding.** Sail passes the required Arrow schema to
   `LanceReader.pruneColumns`; the reader forwards its names to
   `LanceDataset.scanner(columns=...)`. Wide unused payloads never have to be
   produced as Arrow columns. Empty projections preserve batch row counts,
   allowing `COUNT(*)` without decoding payload columns.
3. **SQL predicates inside Lance.** Supported predicates become scanner
   `filter=` SQL, enabling Lance's statistics pruning, scalar indexes, and
   selective payload reads. Indexes are used if present; the plugin does not
   build or mutate them.
4. **Late materialization.** Lance can read filter columns first, then fetch
   wide payloads only for surviving rows. `true` is useful for highly selective
   filters; it is not universally better than early materialization.
5. **Incremental Arrow output.** No `to_table()`, pandas conversion, Python row
   objects, or full-fragment buffering in the read path. The native batch
   reader is closed when the Python generator closes.
6. **Parallel fragment scans.** Tasks contain only version and fragment IDs;
   native datasets/scanners are opened on workers, never pickled. All tasks
   use the same manifest version and honor its deletion files.

The reader pins the latest version **when physical reader planning occurs**,
not when the Python DataFrame is constructed. Appends after planning cannot
leak into later tasks. Explicit `version` gives reproducibility across
separate actions and schema discovery. A schema change between discovery and
planning fails clearly instead of coercing or silently mixing schemas.
Do not vacuum a pinned version while a query is using it.

## Predicate semantics

Supported column/literal types are integer, Boolean, and string, with no
implicit cross-type coercion. Supported operations:

- `=`, `<`, `<=`, `>`, `>=`, null-safe equality, and `NOT`;
- `IS NULL` / `IS NOT NULL`;
- nonempty `IN` lists up to 4096 elements, including SQL null semantics;
- literal starts-with, ends-with, and contains;
- nested struct paths with nonnullable parents when the bridge supplies them.

Independent input filters are conjuncts; supported ones can be pushed even
when another filter remains residual. Unsupported subexpressions are never
discarded under `NOT` or `OR`. If PySpark later exposes `And`/`Or` filter
classes, the translator requires both children to translate exactly.

Identifiers are quoted; names containing backticks are declined because
Lance does not reliably resolve them even when escaped. String literals
escape single quotes. String predicates use functions rather than `LIKE`,
so `%`, `_`, backslashes, and Unicode retain their literal meanings.

The translator deliberately declines:

- float comparisons, including finite literals: Spark's NaN ordering can
  differ from a separately installed Lance/DataFusion;
- date/timestamp comparisons: Sail's current filter bridge turns these
  literals into bare integers and loses the type/timezone information needed
  for safe translation;
- decimals, binary/list literals, casts, arithmetic, and other expressions
  absent from the Python filter API;
- empty or oversized `IN`, unknown columns, and type mismatches.
- paths through nullable structs: older Lance versions do not consistently
  propagate parent validity into child predicates.

Filters are planned without consuming data before being accepted. A Lance
syntax/function rejection returns the original filter to Sail. Errors during
actual reading propagate: the reader never silently drops an accepted filter
or restarts after yielding rows.

Sail currently marks Python filters **Inexact** and rechecks all of them.
The required projection therefore retains residual filter columns, even for
filters this plugin can execute exactly. Rechecking cannot recover rows lost
by an unsafe storage predicate, hence the conservative translation above.

## Comparison with polars-pylance

Investigation sources:

- [Scan implementation](https://github.com/jonasdedden/polars-pylance/blob/main/src/polars_pylance/_scan.py)
- [Expression translator](https://github.com/jonasdedden/polars-pylance/blob/main/src/polars_pylance/_predicate.py)
- [Scanner options](https://github.com/jonasdedden/polars-pylance/blob/main/src/polars_pylance/_options.py)
- [Pushdown guide](https://github.com/jonasdedden/polars-pylance/blob/main/docs/PUSHDOWN.md)
- [Vector search](https://github.com/jonasdedden/polars-pylance/blob/main/docs/VECTOR_SEARCH.md)

| Capability | polars-pylance | This Sail plugin |
| --- | --- | --- |
| Independent compiled Lance wheel | Yes | Yes |
| Projection and streaming Arrow batches | Yes | Yes, with new optional projection hook |
| Scalar indexes, statistics, late materialization | Yes | Yes, delegated to Lance |
| Bounded read-ahead and I/O tuning | Yes | Yes |
| Versioned, fragment-parallel reads | Yes | Yes |
| Column/literal and literal-string filters | Yes | Yes, conservative typed subset |
| Arbitrary original expression access | Serialized Polars IR | No; public PySpark filter objects only |
| Arithmetic, casts, list operations, regex, temporal functions | Many supported | Residual in Sail; not exposed by the bridge |
| Relaxed partial expression pushdown | Safe positive `AND` subsets | Independent conjuncts only |
| Remove exact residual filters | Yes | No; Sail always rechecks |
| Limit/offset pushdown | Yes, with residual safeguards | No; Sail does not forward limits to Python |
| Global top-k vector / full-text search | Yes | Not exposed |
| Generated row IDs / distance / score | Yes | Not exposed |
| Streaming and distributed writes | Yes | Yes, fragment staging plus one commit |

The highest-value follow-up is a **typed expression/capability contract**
between Sail and Python: preserve temporal types and expose a structured
expression tree, negotiate exact versus inexact filters during planning, then
forward safe limits. Avoid serializing Sail's DataFusion SQL wholesale:
different DataFusion versions and Spark semantics make that unsafe.

Vector/full-text options cannot just be passed independently to fragment
tasks: each task would produce a local top-k rather than the dataset-global
result, and ordinary downstream filters are not interchangeable with search
prefilters. A separate search datasource mode with a global scanner and
explicit prefilter semantics is a plausible next step.

## Verification

`test_lance.py` exercises real Lance reads and writes, rejected predicates,
literal escaping, nulls, projections and zero-column row counts, batch sizing,
deletions and unindexed appends with a scalar index, pickling, version
pinning, worker working-directory changes, blobs, staged-file abort cleanup,
and option errors.
An I/O-counter test verifies that both projection and selective late
materialization read fewer payload bytes, rather than merely narrowing
the returned Arrow schema.

`test_spark_*` covers the registered Spark Connect workflow, residual filter
inputs, vector/blob output, actual `columns=` arguments reaching Lance, and a
parallel write/read round trip.
Run in a built Sail environment:

```sh
pytest python/pysail/tests/spark/datasource/test_lance.py
```

The complete plugin test file was checked against the new native bridge with:

- pylance **0.39.0**, PyArrow **24.0.0**, pyspark-client **4.1.3**:
  **81 passed, 1 skipped** (modern blob-materialization integration test);
- pylance **11.0.0**, PyArrow **25.0.1**, pyspark-client **4.2.0**:
  **82 passed**.

The existing Python datasource and Vortex suites also passed with the new
bridge (**54 tests**), including the new positional-load regression test and
verifying the legacy no-pruning-hook path. Rust compile checks, the extension
build, and the new Rust adapter/projection regression tests passed.

This is evidence for those combinations, not a claim that every intervening
release, platform, or Arrow extension type is compatible. These were local
server tests; remote object stores and a multi-machine deployment have not
been exercised.
