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

# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Bundle orchestration export utilities."""

from collections.abc import Mapping, Sequence
import dataclasses
import os
import shutil

from .third_party.neptune.protos import bundle_orchestration_pb2


@dataclasses.dataclass
class SubModel:
  """Defines a unique model in the bundle.

  Attributes:
    name: Name of the sub-model subdirectory in the bundle.
    path: Path to the exported sub-model.
  """

  name: str
  path: str


@dataclasses.dataclass
class PipelineStep:
  """Defines one model's execution within the bundle.

  Attributes:
    model: Name of the sub-model (must match a SubModel.name).
    pipeline: Name of the pipeline in the sub-model to execute.
    repeated_times: Number of times to repeat this step.
    requires_h2d: Whether this step requires Host-to-Device transfer.
    requires_d2h: Whether this step requires Device-to-Host transfer.
  """

  model: str
  pipeline: str
  repeated_times: int = 1


@dataclasses.dataclass
class BundleDefinition:
  """Defines organization of sub-models in the bundle.

  Attributes:
    name: Human-readable name of the bundle.
    version: Version of the bundle.
    pipelines: Maps pipeline name to a sequential list of steps.
  """

  name: str
  version: int
  pipelines: Mapping[str, Sequence[PipelineStep]]


def create_bundle(
    output_path: str,
    bundle_def: BundleDefinition,
    models: Sequence[SubModel],
):
  """Creates the bundle directory, copies sub-models, and writes the proto.

  Args:
    output_path: Path where the bundle should be created.
    bundle_def: Definition of the bundle pipelines and metadata.
    models: Sequence of sub-models to include in the bundle.
  """
  file_utils.mkdir_p(output_path)

  for model in models:
    dest_path = os.path.join(output_path, model.name)
    if os.path.lexists(dest_path):
      if os.path.islink(dest_path):
        os.unlink(dest_path)
      elif os.path.isdir(dest_path):
        shutil.rmtree(dest_path)
      else:
        os.remove(dest_path)
    shutil.copytree(model.path, dest_path)

  proto = bundle_orchestration_pb2.BundlePipelines(
      metadata=bundle_orchestration_pb2.BundleMetadata(
          name=bundle_def.name,
          version=bundle_def.version,
      )
  )

  for pipeline_name, steps in bundle_def.pipelines.items():
    bundle_pipeline = bundle_orchestration_pb2.BundlePipeline()
    for step in steps:
      component = bundle_pipeline.components.add()
      component.model_name = step.model
      component.pipeline_name = step.pipeline
      component.repeated_times = step.repeated_times

    proto.pipelines[pipeline_name].CopyFrom(bundle_pipeline)

  pb_path = os.path.join(output_path, "bundle_orchestration.pb")
  with file_utils.open_file(pb_path, "wb") as f:
    f.write(proto.SerializeToString())
