# Copyright 2026 The Orbax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Defines `SafetensorsLayout`, the handler for Safetensors checkpoints.

Loading is driven by the target sharding: each process resolves which shards of
each tensor its own devices need, maps those shards to byte ranges in the file,
and reads exactly those ranges. There is no cross-process communication and no
XLA compilation -- resharding happens purely through which bytes each process
reads. This mirrors how the v0 array deserializer
(`_src/serialization/serialization.py`) drives reads from
`jax.sharding.Sharding.devices_indices_map`, with hand-computed byte ranges
standing in for TensorStore's chunk index.

Byte runs from every tensor a process needs in one file are coalesced together
by a per-file scheduler under a single bounded-over-read policy: a
`max_over_read_ratio` cap on `block_size / needed_bytes` keeps cross-host
egress from blowing up to N x file_size on inner-dim sharding. The scheduler
then issues each coalesced block as one ranged read -- or streams it in
budget-sized chunks when the block exceeds the in-flight byte budget
(`MemoryOptions.read_concurrent_bytes`, the shared restore memory cap). Peak
host memory beyond the destination shard buffers is bounded at that budget
regardless of block, shard, or file size.
The single-host degenerate case -- one process owns every byte of a file --
collapses to one read per file when the file fits the budget, and a small
number of back-to-back reads otherwise.
"""

import asyncio
import collections
from collections.abc import Awaitable, Callable, Sequence
import dataclasses
import itertools
import json
import math
from typing import Any, cast, NamedTuple

from absl import logging
import humanize
import jax
import jax.numpy as jnp
import numpy as np
from orbax.checkpoint._src.multihost import multihost
from orbax.checkpoint._src.path import async_path
from orbax.checkpoint._src.serialization import limits
from orbax.checkpoint._src.tree import utils as tree_utils
from orbax.checkpoint.experimental.v1._src.context import context as context_lib
from orbax.checkpoint.experimental.v1._src.layout import checkpoint_layout
from orbax.checkpoint.experimental.v1._src.metadata import types as metadata_types
from orbax.checkpoint.experimental.v1._src.path import types

CheckpointLayout = checkpoint_layout.CheckpointLayout
InvalidLayoutError = checkpoint_layout.InvalidLayoutError
Path = types.Path
Checkpointable = checkpoint_layout.Checkpointable
AbstractCheckpointable = checkpoint_layout.AbstractCheckpointable

HEADER_NUM_BYTES = 8
SAFETENSORS_SUFFIX = ".safetensors"

# Per-file scheduler defaults. SafetensorsOptions can override either.
#
# `_DEFAULT_MAX_OVER_READ_RATIO` caps `block_size / needed_bytes` for any
# coalesced read; a host's bytes pulled from one file are bounded at
# `ratio * ideal_bytes`. The default trades up to 1x over-read for fewer
# requests; the row-parallel inner-dim worst case (which would otherwise
# collapse to "read the whole file on every host") is bounded at 2x and the
# partitioner emits more, smaller reads instead.
_DEFAULT_MAX_OVER_READ_RATIO = 2.0
# Always coalesce runs separated by less than this gap, regardless of the
# over-read ratio. Below roughly a page the per-read overhead -- a syscall
# locally, an RTT on object storage -- dwarfs the cost of reading the gap, so a
# strict ratio would otherwise shatter tiny strided reads into many microscopic
# requests. Larger gaps stay governed by `max_over_read_ratio`, where the
# read-count-vs-egress tradeoff is genuinely backend-dependent.
_DEFAULT_MIN_COALESCE_GAP = 4096  # one page
# `_DEFAULT_MAX_IN_FLIGHT_BYTES` caps total bytes the loader holds in flight
# across all concurrent reads in one load call. A single coalesced block
# larger than this budget is streamed in budget-sized chunks; smaller blocks
# run concurrently up to the budget. Peak host memory beyond destination
# shard buffers is bounded at `max_in_flight_bytes` regardless of block,
# shard, or file size.
_DEFAULT_MAX_IN_FLIGHT_BYTES = 2 * 1024 * 1024 * 1024  # 2 GiB

# When the over-read cap fragments a load into many small reads -- the strided
# inner-dim regime -- hint the user toward `max_over_read_ratio`. The warning
# fires only when reads-per-file exceeds this factor *and* the achieved
# over-read stayed near 1x (the cap blocked coalescing, rather than the load
# genuinely spanning many files).
_FRAGMENTED_READ_WARN_FACTOR = 8
_FRAGMENTED_OVER_READ_CEILING = 1.2


def _get_dtypes() -> dict[str, Any]:
  """Returns the mapping from safetensor `dtype` strings to NumPy `dtypes`."""
  return {
      "BOOL": np.bool_,
      "I8": np.int8,
      "U8": np.uint8,
      "I16": np.int16,
      "U16": np.uint16,
      "I32": np.int32,
      "U32": np.uint32,
      "I64": np.int64,
      "U64": np.uint64,
      "F16": np.float16,
      "F32": np.float32,
      "F64": np.float64,
      "BF16": jnp.bfloat16,
      "F8_E5M2": jnp.float8_e5m2,
      "F8_E4M3": jnp.float8_e4m3fn,
  }


def _get_array_properties(info: dict[str, Any]) -> tuple[tuple[int, ...], Any]:
  """Parses shape and dtype from one tensor's safetensors header entry."""
  try:
    dtype = _get_dtypes()[info["dtype"]]
  except KeyError as e:
    raise ValueError(f"Unsupported dtype in SafeTensors header: {e}") from e
  return tuple(info["shape"]), dtype


async def _read_header(path: Path) -> tuple[dict[str, Any], int]:
  """Reads a safetensors header, returning it and the data-section offset."""
  async with async_path.open_file(path, mode="rb") as f:
    if not (size_bytes := cast(bytes, await f.read(HEADER_NUM_BYTES))):
      raise ValueError(f"Could not read header size from {path}.")
    header_size = int.from_bytes(size_bytes, byteorder="little")
    header_bytes = cast(bytes, await f.read(header_size))
    if len(header_bytes) != header_size:
      raise ValueError(f"Could not read header content from {path}.")
    return json.loads(header_bytes), HEADER_NUM_BYTES + header_size


async def _discover_files(path: Path) -> list[Path]:
  """Returns the `.safetensors` file(s) at `path` (a file or a directory)."""
  if await async_path.is_dir(path):
    return sorted(await async_path.glob(path, f"*{SAFETENSORS_SUFFIX}"))
  return [path]


async def _check_file_length(
    path: Path, header: dict[str, Any], data_start: int
) -> None:
  """Fails fast if `path` is too short to hold the tensors its header declares.

  The header records each tensor's `data_offsets` relative to the start of the
  data section (`data_start` bytes in). A truncated or partially-copied file
  still parses, but the byte-range reads that follow would run past EOF and
  surface as a confusing mid-load failure on whichever host owns the missing
  bytes. One `stat` here turns that into a clear up-front error.

  Args:
    path: The `.safetensors` file to check.
    header: The file's parsed JSON header, mapping tensor name to its entry.
    data_start: Byte offset where the data section begins (`8 + header_size`).

  Raises:
    ValueError: If the file is shorter than the data its header declares.
  """
  required = data_start + max(
      (
          info["data_offsets"][1]
          for name, info in header.items()
          if name != "__metadata__"
      ),
      default=0,
  )
  actual = (await async_path.async_stat(path)).length
  if actual < required:
    raise ValueError(
        f"SafeTensors file {path} is truncated: its header declares tensor data"
        f" through byte {required}, but the file is only {actual} bytes."
    )


def _normalize_index(
    index: tuple[Any, ...], global_shape: tuple[int, ...]
) -> tuple[tuple[int, int], ...]:
  """Converts a sharding index (tuple of slices/ints) into (start, stop) pairs."""
  bounds = []
  for dim_index, dim in zip(index, global_shape):
    if isinstance(dim_index, slice):
      if dim_index.step not in (None, 1):
        raise ValueError(f"Strided shard index is unsupported: {dim_index}.")
      start = dim_index.start or 0
      stop = dim if dim_index.stop is None else dim_index.stop
    else:
      start, stop = int(dim_index), int(dim_index) + 1
    bounds.append((start, stop))
  return tuple(bounds)


def _byte_strides(global_shape: tuple[int, ...], itemsize: int) -> list[int]:
  """Row-major byte stride along each dimension."""
  strides = [itemsize] * len(global_shape)
  acc = itemsize
  for i in range(len(global_shape) - 1, -1, -1):
    strides[i] = acc
    acc *= global_shape[i]
  return strides


def index_domain_to_byte_runs(
    bounds: tuple[tuple[int, int], ...],
    global_shape: tuple[int, ...],
    itemsize: int,
    tensor_base: int,
) -> list[tuple[int, int]]:
  """Maps one shard's index domain to `(offset, length)` byte runs in the file.

  A safetensors tensor is a single contiguous row-major blob. The slice of it
  a device owns is contiguous in the file only along trailing dimensions, so a
  shard is a set of equally sized, equally spaced runs: exactly one run when
  the shard splits only the leading dimension (the FSDP / 1-D case), and
  several strided runs when it splits an inner dimension.

  Args:
    bounds: Per-dimension `(start, stop)` of the shard, from `_normalize_index`.
    global_shape: Shape of the whole tensor.
    itemsize: Bytes per element.
    tensor_base: Absolute byte offset of the tensor's first element in the file.

  Returns:
    A list of `(offset, length)` byte runs, sorted by offset.
  """
  if (ndim := len(global_shape)) == 0:  # Scalar.
    return [(tensor_base, itemsize)]
  strides = _byte_strides(global_shape, itemsize)
  full_trailing = 0
  for i in range(ndim - 1, -1, -1):
    if bounds[i] == (0, global_shape[i]):
      full_trailing += 1
    else:
      break
  if (boundary := ndim - full_trailing - 1) < 0:
    return [(tensor_base, strides[0] * global_shape[0])]
  run_length = (bounds[boundary][1] - bounds[boundary][0]) * strides[boundary]
  run_base = tensor_base + bounds[boundary][0] * strides[boundary]
  outer_ranges = [range(bounds[i][0], bounds[i][1]) for i in range(boundary)]
  runs = []
  for outer_index in itertools.product(*outer_ranges):
    offset = run_base
    for dim, coord in enumerate(outer_index):
      offset += coord * strides[dim]
    runs.append((offset, run_length))
  runs.sort()
  return runs


@dataclasses.dataclass(frozen=True)
class _CoalescedBlock:
  """One scheduled read covering several registered runs."""

  start: int  # Absolute byte offset of the read.
  end: int  # Exclusive end byte offset.
  members: tuple[int, ...]  # Indices into the scheduler's request list.


def _partition_runs(
    runs: Sequence[tuple[int, int]],
    max_over_read_ratio: float,
    min_coalesce_gap: int = _DEFAULT_MIN_COALESCE_GAP,
) -> list[_CoalescedBlock]:
  """Greedily partitions `(offset, length)` runs into coalesced read blocks.

  Each emitted block is read with one ranged read (or streamed in budget-sized
  chunks by `_PerFileScheduler` when oversized) and demultiplexed back to its
  member runs. The partitioner absorbs the next run into the current block when
  *either* of two conditions holds:

  - `block_size / needed_bytes <= max_over_read_ratio` -- bounding per-host
    egress amplification. With `max_over_read_ratio == 2.0`, a host pulls at
    most `2x` the bytes it actually needs from this file.
  - the gap to the next run is `<= min_coalesce_gap` -- an absolute floor. The
    ratio test is purely relative, so it would otherwise reject merging tiny
    runs across tiny gaps, shattering strided reads into many microscopic
    requests; below ~a page the per-read overhead dominates and reading the
    gap is always cheaper than a second request.

  Block size is intentionally unbounded here. Peak in-flight memory is bounded
  separately by the scheduler's byte budget (`max_in_flight_bytes`), which
  also enforces correctness for any block too large to read in one go. This
  separation makes the partition policy purely about over-read and the
  budget purely about memory -- each knob expressing one concern.

  The single-host degenerate case -- this process needs every byte of the
  file -- has ratio `1.0` everywhere and collapses to a single block per file.
  FSDP with one contiguous shard per host per file likewise collapses to one
  block per host.

  Args:
    runs: `(offset, length)` byte runs; need not be sorted.
    max_over_read_ratio: Upper bound on `block_size / needed_bytes` per block.
      Must be `>= 1.0`; `1.0` disables any over-read.
    min_coalesce_gap: Runs separated by at most this many bytes are always
      merged, regardless of the ratio. `0` disables the floor (pure ratio).

  Returns:
    A list of `_CoalescedBlock` covering every input run, with each block's
    `members` tuple recording the indices (into the original `runs`) it
    serves. Members are in the order the partitioner encountered them after
    sorting by offset.
  """
  if not runs:
    return []
  if max_over_read_ratio < 1.0:
    raise ValueError(
        f"max_over_read_ratio must be >= 1.0, got {max_over_read_ratio}."
    )
  indexed = sorted(enumerate(runs), key=lambda x: x[1][0])
  blocks: list[_CoalescedBlock] = []
  cur_start: int | None = None
  cur_end = 0
  cur_needed = 0
  cur_members: list[int] = []
  for orig_idx, (offset, length) in indexed:
    end = offset + length
    if cur_start is None:
      cur_start, cur_end = offset, end
      cur_needed = length
      cur_members = [orig_idx]
      continue
    gap = offset - cur_end
    new_end = max(cur_end, end)
    new_size = new_end - cur_start
    new_needed = cur_needed + length
    if gap <= min_coalesce_gap or new_size <= max_over_read_ratio * new_needed:
      cur_end = new_end
      cur_needed = new_needed
      cur_members.append(orig_idx)
    else:
      blocks.append(_CoalescedBlock(cur_start, cur_end, tuple(cur_members)))
      cur_start, cur_end = offset, end
      cur_needed = length
      cur_members = [orig_idx]
  if cur_start is not None:
    blocks.append(_CoalescedBlock(cur_start, cur_end, tuple(cur_members)))
  return blocks


def _replicated_sharding() -> jax.sharding.Sharding:
  """A `NamedSharding` that fully replicates an array across all devices."""
  mesh = jax.sharding.Mesh(np.asarray(jax.devices()), ("_safetensors_replica",))
  return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())


class _FileEntry(NamedTuple):
  """Where to find one tensor in the on-disk checkpoint."""

  path: Path  # The .safetensors file this tensor lives in.
  info: dict[str, Any]  # The tensor's entry from the file's header.
  data_start: int  # Byte offset of the file's data section.


_RunCallback = Callable[[memoryview], None]


class _ReadStats(NamedTuple):
  """Per-file read accounting, aggregated for the fragmentation hint."""

  needed_bytes: int  # Bytes the registered runs actually need.
  read_bytes: int  # Bytes the coalesced blocks pull (>= needed_bytes).
  num_reads: int  # Coalesced blocks (ranged-read requests; ~1 per file ideal).
  num_gets: int  # Actual storage GETs (>= num_reads; splits blocks > budget).


class _PerFileScheduler:
  """Coalesces and issues reads for one safetensors file.

  Callers `register(offset, length, callback)` synchronously while they plan
  their per-tensor reads. After all registrations complete, `execute()`
  partitions the registered runs under the over-read ratio cap, then for
  each coalesced block issues one ranged read -- or streams the block in
  budget-sized chunks if it exceeds `max_in_flight_bytes`. Each chunk's
  bytes are dispatched to every registered callback whose range intersects
  it, sliced to the run-local bytes the callback expects.

  Callback contract: a callback for a registered run may be invoked
  multiple times, always in increasing offset order, each invocation
  carrying a contiguous sub-range of the run's bytes. The callback should
  write the bytes into its destination buffer starting at the cursor
  position equal to "total bytes already delivered to this callback." The
  helpers `_register_*_tensor_reads` in this file follow that pattern.

  Concurrency: `register()` must finish for every tensor that touches this
  file before `execute()` runs; otherwise a late registration would be
  silently dropped. The `SafetensorsLayout._load` orchestration arranges
  this by performing all registrations synchronously before scheduling any
  scheduler `execute()` task. Peak in-flight memory is bounded at the
  shared budget regardless of how many schedulers run concurrently.
  """

  def __init__(
      self,
      path: Path,
      byte_budget: limits.LimitInFlightBytes,
      max_over_read_ratio: float,
  ):
    self._path = path
    self._budget = byte_budget
    self._max_over_read_ratio = max_over_read_ratio
    self._requests: list[tuple[int, int, _RunCallback]] = []

  def register(self, offset: int, length: int, callback: _RunCallback) -> None:
    """Records a byte range this host needs and the callback to receive it."""
    self._requests.append((offset, length, callback))

  async def execute(self) -> _ReadStats:
    """Reads all registered byte runs and returns this file's read accounting."""
    if not self._requests:
      return _ReadStats(needed_bytes=0, read_bytes=0, num_reads=0, num_gets=0)
    runs = [(off, length) for off, length, _ in self._requests]
    blocks = _partition_runs(runs, self._max_over_read_ratio)
    try:
      gets_per_block = await asyncio.gather(
          *[self._read_block(b) for b in blocks]
      )
    finally:
      # Drop callback closures so any buffers they captured can be GC'd as
      # soon as their callers release them.
      self._requests = []
    return _ReadStats(
        needed_bytes=sum(length for _, length in runs),
        read_bytes=sum(b.end - b.start for b in blocks),
        num_reads=len(blocks),
        num_gets=sum(gets_per_block),
    )

  async def _read_block(self, block: _CoalescedBlock) -> int:
    """Reads one coalesced block, streaming in chunks if oversized.

    Args:
      block: The coalesced byte range to read from this file.

    Returns:
      The number of storage GETs issued for the block -- one per chunk, so a
      block within the in-flight budget is a single GET and a larger one is
      several.
    """
    # `LimitInFlightBytes` reserves strictly fewer bytes than its max, and the
    # limiter is sized one over `max_in_flight_bytes` (see `_load`), so the
    # largest reservable chunk is exactly `max_in_flight_bytes`.
    chunk_cap = self._budget.max_bytes - 1
    pos = block.start
    gets = 0
    while pos < block.end:
      chunk_end = min(pos + chunk_cap, block.end)
      chunk_len = chunk_end - pos
      async with limits.reserved_bytes(self._budget, chunk_len):
        async with async_path.open_file(self._path, mode="rb") as f:
          await f.seek(pos)
          # `AsyncFile.read` is declared `bytes | str`; `mode="rb"` guarantees
          # bytes at runtime but pytype can't narrow that statically.
          chunk_bytes = cast(bytes, await f.read(chunk_len))
        mv = memoryview(chunk_bytes)
        self._dispatch_chunk(block, mv, pos, chunk_len)
        # Drop our refs; the buffer can be GC'd once any memoryviews the
        # callbacks may have retained go out of scope.
        del mv, chunk_bytes
      gets += 1
      pos = chunk_end
    return gets

  def _dispatch_chunk(
      self,
      block: _CoalescedBlock,
      mv: memoryview,
      chunk_start: int,
      chunk_len: int,
  ) -> None:
    """Invokes every member's callback with the slice of `mv` it owns.

    For each registered run in `block`, computes the intersection with the
    chunk `[chunk_start, chunk_start + chunk_len)` and -- if non-empty --
    invokes the callback with the corresponding slice of `mv`. Runs that do
    not intersect this chunk are skipped (they will be served by another
    chunk in the same block).

    Args:
      block: The coalesced block whose member runs may intersect the chunk.
      mv: A memoryview over the bytes of the current chunk.
      chunk_start: Absolute file offset where the chunk begins.
      chunk_len: Length of the chunk in bytes.
    """
    chunk_end = chunk_start + chunk_len
    for member_idx in block.members:
      run_offset, run_length, callback = self._requests[member_idx]
      run_end = run_offset + run_length
      hit_start = max(chunk_start, run_offset)
      hit_end = min(chunk_end, run_end)
      if hit_start < hit_end:
        local_start = hit_start - chunk_start
        local_end = hit_end - chunk_start
        callback(mv[local_start:local_end])


def _allocate_buffer(
    shape: tuple[int, ...], dtype: np.dtype
) -> tuple[np.ndarray, np.ndarray]:
  """Allocates a contiguous buffer + a flat `uint8` view over its bytes.

  The flat view is the canonical write-target for byte-level callbacks; it
  works uniformly for scalars (`shape == ()`), 1-D, and N-D buffers, where
  `np.ndarray.view(uint8)` on a 0-D array does not.

  Args:
    shape: Shape of the typed buffer to allocate.
    dtype: Element dtype of the typed buffer.

  Returns:
    A `(buffer, flat_u8)` pair: the typed `shape`-shaped array and a flat
    `uint8` view aliasing the same memory.
  """
  num_elements = math.prod(shape) if shape else 1
  flat_u8 = np.empty(num_elements * dtype.itemsize, dtype=np.uint8)
  buf = flat_u8.view(dtype).reshape(shape)
  return buf, flat_u8


def _make_cursor_receive(dst: np.ndarray) -> _RunCallback:
  """Builds a callback that appends successive chunks into a flat buffer.

  Implements the partial-delivery contract from `_PerFileScheduler`: each
  invocation writes the chunk's bytes at the cursor position and advances
  the cursor by the chunk length. After all expected chunks arrive, the
  cursor equals `len(dst)`.

  Args:
    dst: Flat destination buffer the chunks are written into.

  Returns:
    A callback that writes each received chunk at the running cursor.
  """
  cursor = [0]

  def receive(chunk: memoryview) -> None:
    n = len(chunk)
    end = cursor[0] + n
    dst[cursor[0] : end] = np.frombuffer(chunk, dtype=np.uint8)
    cursor[0] = end

  return receive


def _register_whole_tensor_reads(
    entry: _FileEntry,
    scheduler: _PerFileScheduler,
) -> Callable[[], np.ndarray]:
  """Sync: pre-allocates the destination array and registers one read.

  Used when no `abstract_state` is given -- there is no target sharding to
  drive a partial read, so each tensor is materialised in full on this host.
  The scheduler writes directly into the pre-allocated array; the returned
  builder function just returns it.

  Args:
    entry: The file entry describing the tensor's location and header info.
    scheduler: The per-file read scheduler the read is registered with.

  Returns:
    A builder that returns the fully materialised array after execution.
  """
  shape, dtype = _get_array_properties(entry.info)
  dtype = np.dtype(dtype)
  begin, end = entry.info["data_offsets"]
  buf, flat_u8 = _allocate_buffer(shape, dtype)
  scheduler.register(
      entry.data_start + begin, end - begin, _make_cursor_receive(flat_u8)
  )

  def build() -> np.ndarray:
    return buf

  return build


def _register_sharded_tensor_reads(
    entry: _FileEntry,
    abstract_leaf: Any,
    scheduler: _PerFileScheduler,
) -> Callable[[], jax.Array]:
  """Sync: pre-allocates one shard buffer per local-device shard.

  For each unique index domain owned by one of this process's devices, the
  shard buffer is allocated once. Each of the shard's byte runs is
  registered with a callback that copies its bytes into the buffer at the
  correct row offset. The returned builder turns the populated buffers
  (after `scheduler.execute()`) into a sharded `jax.Array`.

  Args:
    entry: The file entry describing the tensor's location and header info.
    abstract_leaf: The abstract array (shape, dtype, sharding) to build.
    scheduler: The per-file read scheduler the reads are registered with.

  Returns:
    A builder that assembles the populated shard buffers into a `jax.Array`.

  Raises:
    ValueError: If the abstract state's shape disagrees with the checkpoint.
  """
  file_shape, file_dtype = _get_array_properties(entry.info)
  file_dtype = np.dtype(file_dtype)
  tensor_base = entry.data_start + entry.info["data_offsets"][0]

  global_shape = tuple(abstract_leaf.shape)
  out_dtype = np.dtype(abstract_leaf.dtype)
  if global_shape != file_shape:
    raise ValueError(
        f"Shape mismatch: abstract state has {global_shape}, the checkpoint"
        f" has {file_shape}."
    )
  sharding = getattr(abstract_leaf, "sharding", None) or _replicated_sharding()

  this_process = multihost.process_index()
  index_to_devices: dict[tuple[tuple[int, int], ...], list[jax.Device]] = (
      collections.defaultdict(list)
  )
  for device, index in sharding.devices_indices_map(global_shape).items():
    if device.process_index == this_process:
      index_to_devices[_normalize_index(index, global_shape)].append(device)

  shard_specs: list[tuple[np.ndarray, list[jax.Device]]] = []
  for bounds, devices in index_to_devices.items():
    shard_shape = tuple(stop - start for start, stop in bounds)
    shard_buf, shard_flat_u8 = _allocate_buffer(shard_shape, file_dtype)
    runs = index_domain_to_byte_runs(
        bounds, global_shape, file_dtype.itemsize, tensor_base
    )
    # Runs are already sorted by offset; their in-memory destination is the
    # corresponding prefix of `shard_flat_u8` in the same order.
    write_offset = 0
    for off, length in runs:
      end_byte = write_offset + length
      scheduler.register(
          off,
          length,
          _make_cursor_receive(shard_flat_u8[write_offset:end_byte]),
      )
      write_offset = end_byte
    shard_specs.append((shard_buf, devices))

  def build() -> jax.Array:
    arrays: list[jax.Array] = []
    # Consume front-to-back, releasing each shard's host buffer once its data
    # is on-device instead of holding every buffer until the load returns.
    # Popping here also frees the file_dtype original the moment `astype` makes
    # the out_dtype copy, so a conversion's transient overlaps just one shard.
    # `pop(0)` preserves order; the per-process shard count is small.
    while shard_specs:
      shard_buf, devices = shard_specs.pop(0)
      if out_dtype != file_dtype:
        shard_buf = shard_buf.astype(out_dtype)
      for d in devices:
        arrays.append(
            jax.device_put(shard_buf, jax.sharding.SingleDeviceSharding(d))
        )
      del shard_buf
    # `dtype` is required when a process contributes no shards (it cannot be
    # inferred from an empty buffer list).
    return jax.make_array_from_single_device_arrays(
        global_shape, sharding, arrays, dtype=out_dtype
    )

  return build


def _record_read_stats(stats: Sequence[_ReadStats]) -> None:
  """Reports this host's read accounting via `jax.monitoring`.

  Emitted once per load with the per-host totals -- a single scalar per name,
  since a metric collector keyed by name would keep only the last of repeated
  emissions. Mirrors how the TensorStore restore path self-reports
  `/jax/orbax/read/...`, so a benchmark can capture safetensors reads through
  the standard jax.monitoring listener with no dependency on this loader.

  Two read counts are reported: `num_reads` is the coalesced ranged-read
  requests (~1 per file when fully coalesced), and `storage_reads` is the
  actual GETs issued, which is >= `num_reads` because a block larger than the
  in-flight budget is fetched in several chunks.

  Args:
    stats: Per-file read accounting from each scheduler's `execute()`.
  """
  jax.monitoring.record_scalar(
      "/jax/orbax/read/safetensors/bytes_read",
      float(sum(s.read_bytes for s in stats)),
  )
  jax.monitoring.record_scalar(
      "/jax/orbax/read/safetensors/num_reads",
      float(sum(s.num_reads for s in stats)),
  )
  jax.monitoring.record_scalar(
      "/jax/orbax/read/safetensors/storage_reads",
      float(sum(s.num_gets for s in stats)),
  )


def _warn_if_reads_fragmented(
    stats: Sequence[_ReadStats],
    num_files: int,
    max_over_read_ratio_is_default: bool,
) -> None:
  """Logs a one-shot hint when the over-read cap fragmented the load.

  In the strided inner-dim regime the `max_over_read_ratio` cap rejects merges
  and the load degrades into many small ranged reads -- RTT-bound and slow on
  object storage. This surfaces that case so a slow load is diagnosable without
  profiling. It fires once, only on the primary process, and only when the user
  is on the default ratio (someone who set the knob has already weighed the
  tradeoff). The signal is a high reads-per-file count together with an achieved
  over-read near 1x -- exactly "the cap blocked coalescing" rather than "the
  load genuinely spans many files."

  Args:
    stats: Per-file read accounting from each scheduler's `execute()`.
    num_files: Number of distinct files the load read from.
    max_over_read_ratio_is_default: Whether the user left `max_over_read_ratio`
      unset; only then is "raise the ratio" actionable advice.
  """
  if not max_over_read_ratio_is_default or num_files == 0:
    return
  if multihost.process_index() != 0:
    return
  total_needed = sum(s.needed_bytes for s in stats)
  total_read = sum(s.read_bytes for s in stats)
  total_reads = sum(s.num_reads for s in stats)
  if total_needed == 0:
    return
  achieved_over_read = total_read / total_needed
  if (
      total_reads <= _FRAGMENTED_READ_WARN_FACTOR * num_files
      or achieved_over_read >= _FRAGMENTED_OVER_READ_CEILING
  ):
    return
  logging.warning(
      "Safetensors load issued %d ranged reads for %s across %d file(s)"
      " (achieved over-read %.2fx). The target sharding produces strided reads"
      " that the default max_over_read_ratio=%.1f keeps fragmented; raising"
      " SafetensorsOptions.max_over_read_ratio trades bounded over-read for far"
      " fewer reads (notably faster on object storage).",
      total_reads,
      humanize.naturalsize(total_needed, binary=True),
      num_files,
      achieved_over_read,
      _DEFAULT_MAX_OVER_READ_RATIO,
  )


class SafetensorsLayout(CheckpointLayout):
  """Handles checkpoints in the HuggingFace Safetensors format.

  Inherits the abstract methods of :py:class:`~.CheckpointLayout`. Loading is
  resharding-aware: see the module docstring. Saving is not yet supported.
  """

  def __init__(self):
    if multihost.is_pathways_backend():
      raise ValueError(
          "SafetensorsLayout is not supported on Pathways backend."
      )

  async def validate_checkpointables(self, path: Path):
    """Validates that `path` is a SafeTensors file or a directory of them."""
    if await async_path.is_file(path):
      if path.suffix == SAFETENSORS_SUFFIX:
        return
      raise InvalidLayoutError(
          f"Failed to interpret path {path} as a SafeTensors checkpoint. A"
          " SafeTensors checkpoint must be a file with the"
          f" '{SAFETENSORS_SUFFIX}' suffix."
      )
    elif await async_path.is_dir(path):
      files = list(await async_path.glob(path, f"*{SAFETENSORS_SUFFIX}"))
      if not files:
        raise InvalidLayoutError(
            f"Directory {path} does not contain any '{SAFETENSORS_SUFFIX}'"
            " files."
        )
    else:
      raise InvalidLayoutError(
          f"Path {path} is neither a file nor a directory or does not exist."
      )

  async def get_checkpointable_names(  # pyrefly: ignore[bad-override]
      self, path: Path
  ) -> list[str]:
    del path
    return []

  async def validate(
      self, path: Path, checkpointable_name: str | None
  ) -> None:
    """No-op: there is nothing PyTree-specific to validate."""
    del path, checkpointable_name

  async def checkpointables_metadata(
      self, path: Path
  ) -> metadata_types.CheckpointMetadata[dict[str, AbstractCheckpointable]]:
    """A SafeTensors checkpoint is always a flat checkpoint; use `metadata`."""
    raise NotImplementedError(
        "SafetensorsLayout does not support `.checkpointables_metadata`. Use"
        " `.metadata` instead."
    )

  async def metadata(
      self, path: Path, checkpointable_name: str | None
  ) -> metadata_types.CheckpointMetadata[AbstractCheckpointable]:
    """Returns the flat checkpoint metadata for the Safetensors checkpoint."""
    del checkpointable_name  # Safetensors is always flat; name is unused.
    item_metadata: dict[str, jax.ShapeDtypeStruct] = {}
    custom_metadata: dict[str, Any] = {}
    commit_timestamp_nsecs: int | None = None
    for file_path in await _discover_files(path):
      header, _ = await _read_header(file_path)
      stat = await async_path.async_stat(file_path)
      ts = int(stat.mtime)
      if commit_timestamp_nsecs is None or ts > commit_timestamp_nsecs:
        commit_timestamp_nsecs = ts
      for name, info in header.items():
        if name == "__metadata__":
          if info:
            custom_metadata.update(info)
          continue
        if name in item_metadata:
          raise ValueError(f"Duplicate tensor {name} found in multiple files.")
        shape, dtype = _get_array_properties(info)
        item_metadata[name] = jax.ShapeDtypeStruct(shape=shape, dtype=dtype)
    return metadata_types.CheckpointMetadata(
        path=path,
        metadata=item_metadata,
        commit_timestamp_nsecs=commit_timestamp_nsecs,
        custom_metadata=custom_metadata,
    )

  async def load(
      self,
      path: Path,
      checkpointable_name: str | None = None,
      abstract_state: AbstractCheckpointable | None = None,
  ) -> Awaitable[Checkpointable]:
    """Loads a flat PyTree of arrays from a SafeTensors checkpoint.

    Args:
      path: A `.safetensors` file, or a directory containing such files.
      checkpointable_name: Unused; SafeTensors checkpoints expose one
        checkpointable.
      abstract_state: An optional flat dict mapping tensor name to an abstract
        leaf (`jax.ShapeDtypeStruct`). When provided, each leaf's sharding
        drives the load: every process reads only the byte ranges its own
        devices need. When `None`, every tensor is materialised in full on
        this host (single-process only).

    Returns:
      An awaitable resolving to the loaded flat dict of arrays.

    Raises:
      ValueError: If `abstract_state` is provided but is not a flat dict.
    """
    del checkpointable_name
    if abstract_state is not None and not tree_utils.is_flat_dict(
        abstract_state
    ):
      raise ValueError("The PyTree is not a flat dictionary.")
    file_index = await self._build_file_index(path)
    return self._load(file_index, abstract_state)

  async def load_checkpointables(
      self,
      path: Path,
      abstract_checkpointables: dict[str, AbstractCheckpointable] | None = None,
  ) -> Awaitable[dict[str, Checkpointable]]:
    raise NotImplementedError(
        "SafetensorsLayout does not support `.load_checkpointables`. Use"
        " `.load` instead."
    )

  async def save_checkpointables(
      self,
      path: types.PathAwaitingCreation,
      *,
      checkpointables: dict[str, Checkpointable],
  ) -> Awaitable[None]:
    """Raises: saving to Safetensors is not supported."""
    del path, checkpointables
    raise NotImplementedError(
        "Saving to Safetensors format is not supported yet."
    )

  async def _build_file_index(self, path: Path) -> dict[str, _FileEntry]:
    """Maps each tensor name to where to find it (`_FileEntry`)."""
    file_index: dict[str, _FileEntry] = {}
    for file_path in await _discover_files(path):
      header, data_start = await _read_header(file_path)
      await _check_file_length(file_path, header, data_start)
      for name, info in header.items():
        if name == "__metadata__":
          continue
        if name in file_index:
          raise ValueError(f"Duplicate tensor {name} found in multiple files.")
        file_index[name] = _FileEntry(file_path, info, data_start)
    return file_index

  async def _load(
      self,
      file_index: dict[str, _FileEntry],
      abstract_state: dict[str, Any] | None,
  ) -> dict[str, Any]:
    """Background load phase: registers reads, runs schedulers, assembles."""
    context = context_lib.get_context()
    opts = context.safetensors_options
    max_over_read_ratio = (
        opts.max_over_read_ratio
        if opts.max_over_read_ratio is not None
        else _DEFAULT_MAX_OVER_READ_RATIO
    )
    # In-flight read bytes reuse the shared restore knob,
    # `MemoryOptions.read_concurrent_bytes` (the same limit TensorStore restore
    # uses). `None` means "no limit" there, but the streaming/chunk-sizing here
    # needs a finite budget, so fall back to the loader default.
    read_concurrent_bytes = context.memory_options.read_concurrent_bytes
    max_in_flight_bytes = (
        read_concurrent_bytes
        if read_concurrent_bytes is not None
        else _DEFAULT_MAX_IN_FLIGHT_BYTES
    )
    # One shared limiter across every file's scheduler. Peak in-flight memory
    # for the entire load is bounded at `max_in_flight_bytes` regardless of
    # how many files or how large each file is. `LimitInFlightBytes` reserves
    # strictly fewer bytes than its max, so size it one over: a single chunk
    # can then reserve the full `max_in_flight_bytes`, and concurrent
    # reservations still sum to at most `max_in_flight_bytes`.
    byte_budget = limits.LimitInFlightBytes(max_in_flight_bytes + 1)
    schedulers: dict[Path, _PerFileScheduler] = {}

    def scheduler_for(file_path: Path) -> _PerFileScheduler:
      if (sched := schedulers.get(file_path)) is None:
        sched = _PerFileScheduler(file_path, byte_budget, max_over_read_ratio)
        schedulers[file_path] = sched
      return sched

    if abstract_state is None:
      if multihost.process_count() > 1:
        raise ValueError(
            "abstract_state must be provided for multi-process loading."
        )
      names = list(file_index.keys())
    else:
      for name in abstract_state:
        if name not in file_index:
          raise KeyError(
              f"Tensor '{name}' not found in Safetensors checkpoint."
          )
      names = sorted(abstract_state)

    # Phase 1 (sync): register every byte range this process needs with the
    # per-file schedulers and pre-allocate the destination buffers. Each
    # register returns a `build()` that turns the (yet-unfilled) buffers
    # into a `jax.Array` once the schedulers have run.
    build_fns: list[Callable[[], Any]] = []
    for name in names:
      entry = file_index[name]
      sched = scheduler_for(entry.path)
      if abstract_state is None:
        build_fns.append(_register_whole_tensor_reads(entry, sched))
      else:
        build_fns.append(
            _register_sharded_tensor_reads(entry, abstract_state[name], sched)
        )

    # Phase 2 (async): all reads in parallel, bounded by the shared limiter.
    # The scheduler's callbacks fill the pre-allocated buffers in place; no
    # intermediate byte slices are held in futures.
    stats = await asyncio.gather(*[s.execute() for s in schedulers.values()])
    _record_read_stats(stats)
    _warn_if_reads_fragmented(
        stats, len(schedulers), opts.max_over_read_ratio is None
    )

    # Phase 3 (sync): build `jax.Array`s from the now-populated buffers.
    arrays = [fn() for fn in build_fns]
    return dict(zip(names, arrays))
