# Copyright 2024 The Orbax Authors.
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

"""Saving to disk.

This file contains facilities that can save a `Module` to disk.
"""

from collections.abc import Mapping
import dataclasses
import os
from typing import Optional

from absl import logging
from orbax.experimental.model.core.python import file_utils
from orbax.experimental.model.core.python import manifest_constants
from orbax.experimental.model.core.python import module
from orbax.experimental.model.core.python import unstructured_data
from orbax.experimental.model.core.python.concrete_function import ConcreteFunction
from orbax.experimental.model.core.python.concrete_function import Variable
from orbax.experimental.model.core.python.manifest_util import build_manifest_proto
from orbax.experimental.model.core.python.unstructured_data import UnstructuredData
from orbax.experimental.model.core.python.util import object_identity

PolymorphicFunction = module.PolymorphicFunction
Module = module.Module


def named_variables_and_functions(
    m: Module,
) -> tuple[Mapping[Variable, str], Mapping[ConcreteFunction, str]]:
  """Returns all variables and functions in the Module."""
  # TODO(b/328686396): Improve naming. The function names are used to name the
  # SignatureDefs (user-facing). Variable names are not user-facing when serving
  # but would be nice to have readable names for debugging.

  # Maps vars -> names
  named_vars = object_identity.ObjectIdentityDictionary()
  # Maps ConcreteFunction -> string names
  named_functions = object_identity.ObjectIdentityDictionary()
  to_visit = list(m.get_dict().items())
  while to_visit:
    key, child = to_visit.pop()
    if isinstance(child, ConcreteFunction):
      if child in named_functions:
        continue
      named_functions[child] = key

      for i, var in enumerate(child.captured_vars):
        if var in named_vars:
          continue
        named_vars[var] = f'{key}/capture_{i}'
    elif isinstance(child, PolymorphicFunction):
      to_visit.extend(
          (f'{key}/concrete_{i}', fn) for i, fn in enumerate(child.concrete_fns)
      )
    elif isinstance(child, Variable):
      if child not in named_vars:
        named_vars[child] = key
    else:
      raise TypeError(f'Unknown object type: {type(child)}')

  return named_vars, named_functions


@dataclasses.dataclass
class SupplementalInfo:
  """Supplemental info to be saved in the manifest.pb.

  Attributes:
    data: The supplemental info to be saved.
    save_as: An optional relative file name. If set, `data` will be saved as a
      separate file using this file name (and in this case the supplemental info
      mustn't be a `file_location` already).
  """

  data: UnstructuredData

  save_as: str | None


@dataclasses.dataclass
class SaveOptions:
  """Options for saving to FSM SavedModel.

  This function may be used in the `options` argument in functions that
  save an FSM SavedModel.

  Attributes:
    function_aliases: A mapping from user-chosen function alias name to the
      function that runs on TPU.
    version: The serializatioin-format version. With version >= 2 it genertes
      manifest.pb whereas with version <= 1 it generates saved_model.pb .
      Defaults to 1.
    supplemental_info: Optional. An `UnstructuredData` (or a string-map of them)
      to be saved in the manifest.pb as the global supplemental info.
  """

  function_aliases: Mapping[str, ConcreteFunction] | None = None
  version: int | None = None
  supplemental_info: (
      SupplementalInfo | Mapping[str, SupplementalInfo] | None
  ) = None


def _save_single_supplemental(
    supplemental_info: SupplementalInfo,
    path: str,
) -> UnstructuredData:
  """Saves a single supplemental to disk."""
  data = supplemental_info.data
  if supplemental_info.save_as is not None:
    data = unstructured_data.write_inlined_data_to_file(
        data,
        path,
        supplemental_info.save_as,
    )
  return data


def _save_supplementals(
    supplemental_info: SupplementalInfo | Mapping[str, SupplementalInfo] | None,
    path: str,
) -> UnstructuredData | Mapping[str, UnstructuredData] | None:
  """Saves supplementals to disk."""
  if supplemental_info is None:
    return None
  if isinstance(supplemental_info, SupplementalInfo):
    return _save_single_supplemental(supplemental_info, path)
  else:
    return {
        name: _save_single_supplemental(supp, path)
        for name, supp in supplemental_info.items()
    }


def save(
    m: Module,
    path: str,
    options: Optional[SaveOptions] = None,
) -> None:
  """Saved the Module to disk."""
  if options is None:
    raise ValueError('`options` must not be `None` to save version >= 2.')
  assert options.version is not None and options.version >= 2
  logging.info('Save version: %d', options.version)
  # Generate and export the Manifest proto.
  supplemental_info = _save_supplementals(options.supplemental_info, path)
  manifest_proto = build_manifest_proto(
      m,
      path,
      supplemental_info=supplemental_info,
  )
  with file_utils.open_file(
      os.path.join(path, manifest_constants.MANIFEST_FILENAME), 'wb'
  ) as f:
    f.write(manifest_proto.SerializeToString())
