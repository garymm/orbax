# Copyright 2023 The Orbax Authors.
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

import functools
import os

from absl.testing import parameterized
import jax
import jax.numpy as jnp
from orbax.export.export_manager import ExportManager
from orbax.export.export_manager import make_e2e_inference_fn
from orbax.export.jax_module import JaxModule
from orbax.export.serving_config import ServingConfig
from orbax.export.serving_config import TensorSpecWithDefault
import tensorflow as tf


def _from_feature_dict(feature_dict):
  return feature_dict['feat']


def _add_output_name(outputs):
  return {'outputs': outputs}


_ZERO_VAR = tf.Variable(0)


def _add_zero(x):
  return x + _ZERO_VAR


def _linear(params, x, with_bias=False):
  y = x @ params['w']
  if with_bias:
    return y + params['b']
  return y


class ExportManagerTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._output_dir = self.create_tempdir().full_path

  @parameterized.named_parameters(
      dict(
          testcase_name='normal',
          input_signature=[
              {'feat': tf.TensorSpec((), tf.dtypes.int32, 'feat')}
          ],
          preprocessor=_from_feature_dict,
          postprocessor=_add_output_name,
          inputs={'feat': tf.constant(1)},
          outputs={'outputs': tf.constant(2)},
      ),
      dict(
          testcase_name='embedded input signature',
          preprocessor=tf.function(
              _from_feature_dict,
              [{'feat': tf.TensorSpec((), tf.dtypes.int32, 'feat')}],
          ),
          postprocessor=_add_output_name,
          inputs={'feat': tf.constant(1)},
          outputs={'outputs': tf.constant(2)},
      ),
      dict(
          testcase_name='no preprocessor',
          input_signature=[tf.TensorSpec((), tf.dtypes.int32, 'feat')],
          postprocessor=_add_output_name,
          inputs=tf.constant(1),
          outputs={'outputs': tf.constant(2)},
      ),
      dict(
          testcase_name='no postprocessor',
          input_signature=[
              {'feat': tf.TensorSpec((), tf.dtypes.int32, 'feat')}
          ],
          preprocessor=_from_feature_dict,
          inputs={'feat': tf.constant(1)},
          outputs=tf.constant(2),
      ),
      dict(
          testcase_name='core module only',
          input_signature=[tf.TensorSpec((), tf.dtypes.int32, 'feat')],
          inputs=tf.constant(1),
          outputs=tf.constant(2),
      ),
      dict(
          testcase_name='default value',
          input_signature=[
              TensorSpecWithDefault(
                  tf.TensorSpec((), tf.dtypes.int32, 'feat'), 1
              )
          ],
          inputs=[],
          outputs=tf.constant(2),
      ),
  )
  def test_make_e2e_inference_fn(
      self,
      inputs,
      outputs,
      input_signature=None,
      preprocessor=None,
      postprocessor=None,
  ):
    method = JaxModule(
        {
            'bias': jnp.array(1)
        },
        lambda p, x: x + p['bias'],
    ).methods[JaxModule.DEFAULT_METHOD_KEY]
    inference_fn = make_e2e_inference_fn(
        method,
        ServingConfig('key', input_signature, preprocessor, postprocessor),
    )
    self.assertAllEqual(inference_fn(inputs), outputs)

  @parameterized.named_parameters(
      dict(
          testcase_name='multiple signatures',
          serving_configs=[
              ServingConfig(
                  'with_processors',
                  input_signature=[{
                      'feat': tf.TensorSpec((), tf.dtypes.int32, 'feat')
                  }],
                  tf_preprocessor=_from_feature_dict,
                  tf_postprocessor=_add_output_name,
              ),
              ServingConfig(
                  'without_processors',
                  input_signature=[tf.TensorSpec((), tf.dtypes.int32)],
              ),
          ],
          expected_keys=['with_processors', 'without_processors']),
      dict(
          testcase_name='multiple keys same signature',
          serving_configs=[
              ServingConfig(
                  ['serving_default', 'without_processors'],
                  input_signature=[tf.TensorSpec((), tf.dtypes.int32)],
              ),
          ],
          expected_keys=['serving_default', 'without_processors']),
      dict(
          testcase_name='trackables in preprocessor',
          serving_configs=[
              ServingConfig(
                  'serving_default',
                  input_signature=[tf.TensorSpec((), tf.dtypes.int32)],
                  tf_preprocessor=_add_zero,
                  extra_trackable_resources=_ZERO_VAR,
              ),
          ],
          expected_keys=['serving_default']),
  )
  def test_save(self, serving_configs, expected_keys):
    em = ExportManager(
        JaxModule({'bias': jnp.array(1)}, lambda p, x: x + p['bias']),
        serving_configs,
    )
    em.save(self._output_dir)
    loaded = tf.saved_model.load(self._output_dir, ['serve'])
    self.assertCountEqual(expected_keys, em.serving_signatures.keys())
    self.assertCountEqual(expected_keys, loaded.signatures.keys())

  @parameterized.named_parameters(
      dict(
          testcase_name='all default',
          serving_config=ServingConfig(
              'serving_default',
              input_signature=[
                  TensorSpecWithDefault(tf.TensorSpec((), tf.int32, 'x'), 2),
                  TensorSpecWithDefault(tf.TensorSpec((), tf.int32, 'y'), 3),
              ],
              tf_preprocessor=lambda x, y: x + y,
          ),
          serving_inputs={},
          expected_outputs=6,
      ),
      dict(
          testcase_name='some default',
          serving_config=ServingConfig(
              'serving_default',
              input_signature=[
                  TensorSpecWithDefault(tf.TensorSpec((), tf.int32, 'x'), 2),
                  tf.TensorSpec((), tf.int32, 'y'),
              ],
              tf_preprocessor=lambda x, y: x + y,
          ),
          serving_inputs={'y': 3},
          expected_outputs=6,
      ),
      dict(
          testcase_name='override default',
          serving_config=ServingConfig(
              'serving_default',
              input_signature=[
                  TensorSpecWithDefault(tf.TensorSpec((), tf.int32, 'x'), 2),
                  tf.TensorSpec((), tf.int32, 'y'),
              ],
              tf_preprocessor=lambda x, y: x + y,
          ),
          serving_inputs={'x': 1, 'y': 3},
          expected_outputs=5,
      ),
      dict(
          testcase_name='nested',
          serving_config=ServingConfig(
              'serving_default',
              input_signature=[
                  tf.TensorSpec((), tf.int32, 'x'),
                  {
                      'y': TensorSpecWithDefault(
                          tf.TensorSpec((), tf.int32, 'y'), 2
                      ),
                      'z': TensorSpecWithDefault(
                          tf.TensorSpec((), tf.int32, 'z'), 3
                      ),
                  },
              ],
              tf_preprocessor=lambda x, extra: x + extra['y'] + extra['z'],
          ),
          serving_inputs={'x': 1},
          expected_outputs=7,
      ),
  )
  def test_save_default_inputs(
      self, serving_config, serving_inputs, expected_outputs
  ):
    em = ExportManager(
        JaxModule(
            {'bias': jnp.array(1, jnp.int32)}, lambda p, x: x + p['bias']
        ),
        [serving_config],
    )
    em.save(self._output_dir)
    # TODO(b/277814477): use the TF2 API
    # loaded.signature['serving_default'](**serving_inputs)
    # once it supports default values.
    with tf.compat.v1.Graph().as_default(), tf.compat.v1.Session() as sess:
      meta_graph_def = tf.compat.v1.saved_model.loader.load(
          sess, ['serve'], self._output_dir
      )
      signature_def = meta_graph_def.signature_def[serving_config.signature_key]
      output_tensor_name = signature_def.outputs['output_0'].name
      fetch = sess.graph.get_tensor_by_name(output_tensor_name)
      feed_dict = {
          sess.graph.get_tensor_by_name(signature_def.inputs[k].name): v
          for k, v in serving_inputs.items()
      }
      outputs = sess.run(fetch, feed_dict=feed_dict)
    self.assertAllEqual(outputs, expected_outputs)

  def test_save_multiple_model_functions(self):
    linear_mdl = JaxModule(
        params={
            'w': jnp.zeros((4, 2), jnp.int32),
            'b': jnp.ones((2,), jnp.int32),
        },
        apply_fn={
            'with_bias': functools.partial(_linear, with_bias=True),
            'without_bias': functools.partial(_linear, with_bias=False),
        },
        input_polymorphic_shape={
            'with_bias': None,
            'without_bias': None
        })

    em = ExportManager(
        linear_mdl,
        serving_configs=[
            ServingConfig(
                'serving_default',
                method_key='with_bias',
                input_signature=[
                    tf.TensorSpec(shape=(1, 4), dtype=tf.int32, name='x')
                ],
                tf_postprocessor=lambda out: {'y': out}),
            ServingConfig(
                'no_bias',
                method_key='without_bias',
                input_signature=[
                    tf.TensorSpec(shape=(1, 4), dtype=tf.int32, name='x')
                ],
                tf_postprocessor=lambda out: {'y': out})
        ])
    em.save(self._output_dir)
    loaded = tf.saved_model.load(self._output_dir, ['serve'])

    expected_keys = ['serving_default', 'no_bias']
    self.assertCountEqual(expected_keys, em.serving_signatures.keys())
    self.assertCountEqual(expected_keys, loaded.signatures.keys())

    x = jnp.zeros((1, 4), jnp.int32)
    self.assertAllEqual(loaded.signatures['serving_default'](x=x)['y'],
                        jnp.ones((1, 2)))
    self.assertAllEqual(loaded.signatures['no_bias'](x=x)['y'], jnp.zeros(
        (1, 2)))

  def test_callable_module(self):
    jax_module = JaxModule(
        jnp.asarray(0.0),
        lambda w, x: w + jnp.sum(x['a']['b']),
    )
    dummy_inputs = {'a': {'b': jnp.ones(3, jnp.float32)}}

    input_signature = jax.tree_map(
        lambda x: tf.TensorSpec(dtype=x.dtype, shape=x.shape), dummy_inputs
    )
    em = ExportManager(
        jax_module,
        [
            ServingConfig(
                'serving_default',
                input_signature=[input_signature],
                tf_postprocessor=lambda out: {'y': out},
            )
        ],
    )
    em.save(self._output_dir)
    loaded = em.load(self._output_dir)
    result = loaded(dummy_inputs)
    self.assertAllClose(result['y'], jnp.asarray(3.0))

  def test_save_non_differentiable_fn(self):

    def non_differetiable_fn(_, x):
      _, x = jax.lax.while_loop(
          cond_fun=lambda state: state[0],
          body_fun=lambda state: (False, state[1] + 1),
          init_val=(False, x))
      return x

    serving_config = ServingConfig('serving', [tf.TensorSpec((), tf.float32)])

    # https://github.com/google/jax/blob/main/jax/experimental/jax2tf/README.md#saved-model-for-non-differentiable-jax-functions
    with self.assertRaises(ValueError):
      ExportManager(
          JaxModule({'dummy': jnp.array(1.0)},
                    non_differetiable_fn,
                    trainable=True),
          [serving_config]).save(os.path.join(self._output_dir, '0'))

    # Okay with with_gradients=False (default).
    ExportManager(
        JaxModule({'dummy': jnp.array(1.0)}, non_differetiable_fn),
        [serving_config]).save(os.path.join(self._output_dir, '1'))

  def test_init_invalid_arguments(self):
    single_fn_module = JaxModule({}, lambda p, x: x)
    multi_fn_module = JaxModule({}, {
        'foo': lambda p, x: x,
        'bar': lambda p, x: x
    }, input_polymorphic_shape={'foo': None, 'bar': None})
    with self.assertRaisesRegex(ValueError, 'Duplicated key'):
      ExportManager(single_fn_module, [
          ServingConfig('serving', [tf.TensorSpec((), tf.float32)]),
          ServingConfig('serving', [tf.TensorSpec((), tf.int32)]),
      ])
    with self.assertRaisesRegex(ValueError, 'Duplicated key'):
      ExportManager(single_fn_module, [
          ServingConfig(['serve', 'serve'], [tf.TensorSpec((), tf.float32)]),
      ])
    with self.assertRaisesRegex(ValueError, '`method_key` is not specified'):
      ExportManager(multi_fn_module, [
          ServingConfig('serving', [tf.TensorSpec((), tf.float32)]),
      ])
    with self.assertRaisesRegex(ValueError, 'Method key "baz" is not found'):
      ExportManager(multi_fn_module, [
          ServingConfig(
              'serving', [tf.TensorSpec((), tf.float32)], method_key='baz'),
      ])


if __name__ == '__main__':
  tf.test.main()
