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

"""Saving tests."""

from collections.abc import Sequence
import os
from typing import Tuple

import jax
from jax import export as jax_export
import jax.numpy as jnp
import numpy as np
from orbax.experimental.model.core.protos.saved_model import types_pb2
from orbax.experimental.model.core.python import concrete_function
from orbax.experimental.model.core.python import module
from orbax.experimental.model.core.python import save as save_lib
from orbax.experimental.model.core.python import signature
from orbax.experimental.model.core.python.concrete_function import dtype_from_np_dtype
from orbax.experimental.model.core.python.function import np_dtype_to_shlo_dtype
from orbax.experimental.model.core.python.function import ShloTensorSpec
from orbax.experimental.model.core.python.shlo_function import ShloFunction
import tensorflow as tf

from absl.testing import absltest

save = save_lib.save
Tensor = concrete_function.Tensor
Function = concrete_function.ConcreteFunction
Variable = concrete_function.Variable
TensorSpec = signature.TensorSpec


def read_checkpoint_values(
    prefix: str,
) -> dict[str, tuple[np.ndarray, types_pb2.DataType]]:
  loaded = tf.train.load_checkpoint(prefix)
  contents = {}
  for key in loaded.get_variable_to_dtype_map().keys():
    contents[key] = loaded.get_tensor(key)
  return contents


def jax_spec_from_aval(x: jax.core.AbstractValue) -> jax.ShapeDtypeStruct:
  assert isinstance(x, jax.core.ShapedArray)
  return jax.ShapeDtypeStruct(shape=x.shape, dtype=x.dtype)


def jax_spec_to_shlo_spec(
    jax_spec: Sequence[jax.ShapeDtypeStruct],
) -> Tuple[ShloTensorSpec, ...]:
  return tuple(
      ShloTensorSpec(shape=x.shape, dtype=np_dtype_to_shlo_dtype(x.dtype))
      for x in jax_spec
  )


def jax_spec_to_tensor_spec(x: jax.ShapeDtypeStruct) -> TensorSpec:
  return TensorSpec(shape=x.shape, dtype=dtype_from_np_dtype(x.dtype))


class SaveTest(googletest.TestCase):

  def test_save(self):
    @jax.jit
    def f(xy, captured):
      return captured + xy, captured - xy

    myvar = jnp.array(np.ones((5, 10), dtype=np.float32))
    jax_in_spec = (
        jax.ShapeDtypeStruct(shape=[], dtype=myvar.dtype),
        jax.ShapeDtypeStruct(shape=myvar.shape, dtype=myvar.dtype),
    )
    exported = jax_export.export(f)(*jax_in_spec)
    jax_out_spec = tuple(jax_spec_from_aval(x) for x in exported.out_avals)

    input_signature = (jax_spec_to_tensor_spec(jax_in_spec[0]),)
    output_signature = tuple(jax_spec_to_tensor_spec(x) for x in jax_out_spec)
    func = Function(
        input_signature=input_signature,
        output_signature=output_signature,
        base_fn=ShloFunction(
            input_signature=jax_spec_to_shlo_spec(jax_in_spec),
            output_signature=jax_spec_to_shlo_spec(jax_out_spec),
            mlir_module_serialized=exported.mlir_module_serialized,
            calling_convention_version=exported.calling_convention_version,
            module_kept_var_idx=exported.module_kept_var_idx,
            lowering_platforms=exported.platforms,
            supplemental_info=None,
            physical_in_dtypes=(None,) * len(jax_in_spec),
            physical_out_dtypes=(None,) * len(jax_out_spec),
        ),
        captured_vars=(Variable(Tensor(np.asarray(myvar))),),
    )

    m = module.Module()
    m.add_variables(
        'a', concrete_function.Variable(Tensor(np.array([1, 2, 3, 4, 5])))
    )
    m.add_variables(
        'b', concrete_function.Variable(Tensor(np.array([1, 25, 34, 14, 20])))
    )
    m.add_concrete_function('add', func)

    save_path = os.path.join(self.create_tempdir())
    save(m, save_path)

    # Validate exported model using tf.saved_model.load.
    # TODO(b/328687975): loaded.variables will be empty, but the function can be
    # called and the variables can be found in the checkpoint.
    loaded = tf.saved_model.load(save_path)

    inp = np.array(33.4, dtype=np.float32)
    out = loaded.signatures['add'](tf.constant(inp))
    expected = f(inp, myvar)

    np.testing.assert_array_equal(expected[0], out['output_0'])
    np.testing.assert_array_equal(expected[1], out['output_1'])

    ckpt = read_checkpoint_values(
        os.path.join(save_path, 'variables', 'variables')
    )
    self.assertEqual(ckpt.keys(), {'a', 'b', 'add/capture_0'})
    np.testing.assert_array_equal(ckpt['a'], np.array([1, 2, 3, 4, 5]))
    np.testing.assert_array_equal(ckpt['b'], np.array([1, 25, 34, 14, 20]))
    np.testing.assert_array_equal(ckpt['add/capture_0'], myvar)


if __name__ == '__main__':
  googletest.main()