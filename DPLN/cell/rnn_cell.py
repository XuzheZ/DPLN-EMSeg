# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Module for constructing RNN Cells. These codes are derived from tf.nn.rnn_cell"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math

from tensorflow.contrib.compiler import jit
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.contrib.rnn.python.ops import core_rnn_cell
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import op_def_registry
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import base as base_layer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_impl  # pylint: disable=unused-import
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import partitioned_variables  # pylint: disable=unused-import
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest


def _get_concat_variable(name, shape, dtype, num_shards):
  """Get a sharded variable concatenated into one tensor."""
  sharded_variable = _get_sharded_variable(name, shape, dtype, num_shards)
  if len(sharded_variable) == 1:
    return sharded_variable[0]

  concat_name = name + "/concat"
  concat_full_name = vs.get_variable_scope().name + "/" + concat_name + ":0"
  for value in ops.get_collection(ops.GraphKeys.CONCATENATED_VARIABLES):
    if value.name == concat_full_name:
      return value

  concat_variable = array_ops.concat(sharded_variable, 0, name=concat_name)
  ops.add_to_collection(ops.GraphKeys.CONCATENATED_VARIABLES, concat_variable)
  return concat_variable


def _get_sharded_variable(name, shape, dtype, num_shards):
  """Get a list of sharded variables with the given dtype."""
  if num_shards > shape[0]:
    raise ValueError("Too many shards: shape=%s, num_shards=%d" % (shape,num_shards))
  unit_shard_size = int(math.floor(shape[0] / num_shards))
  remaining_rows = shape[0] - unit_shard_size * num_shards

  shards = []
  for i in range(num_shards):
    current_size = unit_shard_size
    if i < remaining_rows:
      current_size += 1
    shards.append(
        vs.get_variable(
            name + "_%d" % i, [current_size] + shape[1:], dtype=dtype))
  return shards


def _norm(g, b, inp, scope):
  shape = inp.get_shape()[-1:]
  gamma_init = init_ops.constant_initializer(g)
  beta_init = init_ops.constant_initializer(b)
  with vs.variable_scope(scope):
    # Initialize beta and gamma for use by layer_norm.
    vs.get_variable("gamma", shape=shape, initializer=gamma_init)
    vs.get_variable("beta", shape=shape, initializer=beta_init)
  normalized = layers.layer_norm(inp, reuse=True, scope=scope)
  return normalized



class ConvLSTMCell(rnn_cell_impl.RNNCell):
  """Convolutional LSTM recurrent network cell, we added "reuse" item.

  https://arxiv.org/pdf/1506.04214v1.pdf
  """

  def __init__(self,
               conv_ndims,
               input_shape,
               output_channels,
               kernel_shape,
               use_bias=True,
               skip_connection=False,
               forget_bias=1.0,
               initializers=None,
               reuse=None):
    """Construct ConvLSTMCell.

    Args:
      conv_ndims: Convolution dimensionality (1, 2 or 3).
      input_shape: Shape of the input as int tuple, excluding the batch size.
      output_channels: int, number of output channels of the conv LSTM.
      kernel_shape: Shape of kernel as in tuple (of size 1,2 or 3).
      use_bias: (bool) Use bias in convolutions.
      skip_connection: If set to `True`, concatenate the input to the
        output of the conv LSTM. Default: `False`.
      forget_bias: Forget bias.
      initializers: Unused.
      name: Name of the module.

    Raises:
      ValueError: If `skip_connection` is `True` and stride is different from 1
        or if `input_shape` is incompatible with `conv_ndims`.
    """
    super(ConvLSTMCell, self).__init__(_reuse=reuse)

    if conv_ndims != len(input_shape) - 1:
      raise ValueError("Invalid input_shape {} for conv_ndims={}.".format(
          input_shape, conv_ndims))

    self._conv_ndims = conv_ndims
    self._input_shape = input_shape
    self._output_channels = output_channels
    self._kernel_shape = kernel_shape
    self._use_bias = use_bias
    self._forget_bias = forget_bias
    self._skip_connection = skip_connection
    self._reuse = reuse

    self._total_output_channels = output_channels
    if self._skip_connection:
      self._total_output_channels += self._input_shape[-1]

    state_size = tensor_shape.TensorShape(
        self._input_shape[:-1] + [self._output_channels])
    self._state_size = rnn_cell_impl.LSTMStateTuple(state_size, state_size)
    self._output_size = tensor_shape.TensorShape(
        self._input_shape[:-1] + [self._total_output_channels])

  @property
  def output_size(self):
    return self._output_size

  @property
  def state_size(self):
    return self._state_size

  def call(self, inputs, state, scope=None):
    with vs.variable_scope(scope, reuse=self._reuse):
        cell, hidden = state
        new_hidden = _conv([inputs, hidden], self._kernel_shape,
                           4 * self._output_channels, self._use_bias)
        gates = array_ops.split(
            value=new_hidden, num_or_size_splits=4, axis=self._conv_ndims + 1)
        input_gate, new_input, forget_gate, output_gate = gates
        new_cell = math_ops.sigmoid(forget_gate + self._forget_bias) * cell
        new_cell += math_ops.sigmoid(input_gate) * math_ops.tanh(new_input)
        output = math_ops.tanh(new_cell) * math_ops.sigmoid(output_gate)

        if self._skip_connection:
          output = array_ops.concat([output, inputs], axis=-1)
        new_state = rnn_cell_impl.LSTMStateTuple(new_cell, output)
        return output, new_state


class Conv1DLSTMCell(ConvLSTMCell):
  """1D Convolutional LSTM recurrent network cell.

  https://arxiv.org/pdf/1506.04214v1.pdf
  """

  def __init__(self, name="conv_1d_lstm_cell", **kwargs):
    """Construct Conv1DLSTM. See `ConvLSTMCell` for more details."""
    super(Conv1DLSTMCell, self).__init__(conv_ndims=1, name=name, **kwargs)


class Conv2DLSTMCell(ConvLSTMCell):
  """2D Convolutional LSTM recurrent network cell.

  https://arxiv.org/pdf/1506.04214v1.pdf
  """

  def __init__(self, name="conv_2d_lstm_cell", **kwargs):
    """Construct Conv2DLSTM. See `ConvLSTMCell` for more details."""
    super(Conv2DLSTMCell, self).__init__(conv_ndims=2, name=name, **kwargs)


class Conv3DLSTMCell(ConvLSTMCell):
  """3D Convolutional LSTM recurrent network cell.

  https://arxiv.org/pdf/1506.04214v1.pdf
  """

  def __init__(self, name="conv_3d_lstm_cell", **kwargs):
    """Construct Conv3DLSTM. See `ConvLSTMCell` for more details."""
    super(Conv3DLSTMCell, self).__init__(conv_ndims=3, name=name, **kwargs)


def _conv(args, filter_size, num_features, bias, bias_start=0.0):
  """Convolution.

  Args:
    args: a Tensor or a list of Tensors of dimension 3D, 4D or 5D,
    batch x n, Tensors.
    filter_size: int tuple of filter height and width.
    num_features: int, number of features.
    bias: Whether to use biases in the convolution layer.
    bias_start: starting value to initialize the bias; 0 by default.

  Returns:
    A 3D, 4D, or 5D Tensor with shape [batch ... num_features]

  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  """

  # Calculate the total size of arguments on dimension 1.
  total_arg_size_depth = 0
  shapes = [a.get_shape().as_list() for a in args]
  shape_length = len(shapes[0])
  for shape in shapes:
    if len(shape) not in [3, 4, 5]:
      raise ValueError("Conv Linear expects 3D, 4D "
                       "or 5D arguments: %s" % str(shapes))
    if len(shape) != len(shapes[0]):
      raise ValueError("Conv Linear expects all args "
                       "to be of same Dimension: %s" % str(shapes))
    else:
      total_arg_size_depth += shape[-1]
  dtype = [a.dtype for a in args][0]

  # determine correct conv operation
  if shape_length == 3:
    conv_op = nn_ops.conv1d
    strides = 1
  elif shape_length == 4:
    conv_op = nn_ops.conv2d
    strides = shape_length * [1]
  elif shape_length == 5:
    conv_op = nn_ops.conv3d
    strides = shape_length * [1]

  # Now the computation.
  kernel = vs.get_variable(
      "kernel", filter_size + [total_arg_size_depth, num_features], dtype=dtype)
  if len(args) == 1:
    res = conv_op(args[0], kernel, strides, padding="SAME")
  else:
    res = conv_op(
        array_ops.concat(axis=shape_length - 1, values=args),
        kernel,
        strides,
        padding="SAME")
  if not bias:
    return res
  bias_term = vs.get_variable(
      "biases", [num_features],
      dtype=dtype,
      initializer=init_ops.constant_initializer(bias_start, dtype=dtype))
  return res + bias_term
