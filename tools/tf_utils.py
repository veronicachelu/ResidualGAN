import collections
import logging
import logging
import os
import re
import ruamel.yaml as yaml
import tensorflow as tf
import tensorflow.contrib.layers as layers


def lrelu(x, alpha=0.2, name="LeakyReLU"):
  return tf.maximum(x, alpha * x, name=name)


def gradient_summaries(grad_vars, scope='gradients'):
  summaries = []
  for grad, var in grad_vars:
    if grad is None:
      continue
    summaries.append(tf.summary.histogram(scope + '/' + var.name + '_grad', grad))
    summaries.append(tf.summary.histogram(scope + '/' + var.name, var))
  return tf.summary.merge(summaries)


def save_config(config, logdir=None):
  if logdir:
    with config.unlocked:
      config.logdir = logdir
    message = 'Start a new run and write summaries and checkpoints to {}.'
    tf.logging.info(message.format(config.logdir))
    tf.gfile.MakeDirs(config.logdir)
    config_path = os.path.join(config.logdir, 'config.yaml')
    with tf.gfile.FastGFile(config_path, 'w') as file_:
      yaml.dump(config, file_, default_flow_style=False)
  else:
    message = (
      'Start a new run without storing summaries and checkpoints since no '
      'logging directory was specified.')
    tf.logging.info(message)
  return config


def load_config(logdir):
  config_path = logdir and os.path.join(logdir, 'config.yaml')
  if not config_path or not tf.gfile.Exists(config_path):
    message = (
      'Cannot resume an existing run since the logging directory does not '
      'contain a configuration file.')
    raise IOError(message)
  with tf.gfile.FastGFile(config_path, 'r') as file_:
    config = yaml.load(file_)
  message = 'Resume run and write summaries and checkpoints to {}.'
  tf.logging.info(message.format(config.logdir))
  return config


def set_up_logging():
  tf.logging.set_verbosity(tf.logging.INFO)
  logging.getLogger('tensorflow').propagate = False


def define_saver(exclude=None):
  variables = []
  exclude = exclude or []
  exclude = [re.compile(regex) for regex in exclude]
  for variable in tf.global_variables():
    if any(regex.match(variable.name) for regex in exclude):
      continue
    variables.append(variable)
  saver = tf.train.Saver(variables, keep_checkpoint_every_n_hours=5)
  return saver


def upsample(incoming, stride, kernel_size, pad, nb_kernels, weights_initializer, variables_collection, nblayer,
             batch_norm_decay, is_training):
  scope = "deconv_{}".format(nblayer)
  incoming = layers.conv2d(incoming, num_outputs=nb_kernels, kernel_size=kernel_size,
                           stride=1, activation_fn=None,
                           weights_initializer=weights_initializer,
                           padding="VALID" if pad == 0 else "SAME",
                           variables_collections=tf.get_collection(variables_collection),
                           outputs_collections="activations", scope=scope)
  scope = "bn_{}".format(nblayer)
  incoming = tf.contrib.layers.batch_norm(incoming,
                                          center=True, scale=True, decay=batch_norm_decay,
                                          is_training=is_training,
                                          scope=scope)
  incoming = lrelu(incoming)
  input_shape = incoming.get_shape().as_list()
  # with tf.name_scope(scope) as scope:
  incoming = tf.image.resize_nearest_neighbor(
    incoming, size=input_shape[1:3] * tf.constant(stride))
  incoming.set_shape((None, input_shape[1] * stride,
                      input_shape[2] * stride, None))

  return incoming


def dense_upsample(incoming, stride, kernel_size, pad, nb_kernels, weights_initializer, variables_collection, nblayer,
                   batch_norm_decay, is_training):
  scope = "deconv_{}".format(nblayer)
  incoming = layers.conv2d(incoming, num_outputs=(stride ** 2) * nb_kernels, kernel_size=kernel_size,
                           stride=1, activation_fn=None,
                           weights_initializer=weights_initializer,
                           padding="VALID" if pad == 0 else "SAME",
                           variables_collections=tf.get_collection(variables_collection),
                           outputs_collections="activations", scope=scope)
  scope = "bn_{}".format(nblayer)
  incoming = tf.contrib.layers.batch_norm(incoming,
                                          center=True, scale=True, decay=batch_norm_decay,
                                          is_training=is_training,
                                          scope=scope)
  incoming = lrelu(incoming)

  incoming = tf.depth_to_space(incoming, stride, name=None)

  return incoming
