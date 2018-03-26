import os
import tensorflow as tf
import tools
from tools import tf_utils
import configs
import dataset
FLAGS = tf.app.flags.FLAGS


def train(config, logdir):
  tf.reset_default_graph()
  sess = tf.Session()
  with sess:
    # with tf.device("/gpu:0"):
    with config.unlocked:
      config.logdir = logdir
      config.network_optimizer = getattr(tf.train, config.network_optimizer)
      global_step = tf.Variable(1, dtype=tf.int32, name='global_step', trainable=False)

      model = config.model(config, global_step, dataset.get_input(FLAGS.celeba_path, FLAGS.batch_size), FLAGS.train, FLAGS.nb_summaries_outputs)

      if FLAGS.resume:
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(os.path.join(FLAGS.load_from, "models"))
        print("Loading Model from {}".format(ckpt.model_checkpoint_path))
        model.loader.restore(sess, ckpt.model_checkpoint_path)
        sess.run(tf.local_variables_initializer())
      else:
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

      model.train(sess)

def generate(config, logdir):
  tf.reset_default_graph()
  sess = tf.Session()
  with sess:
    # with tf.device("/gpu:0"):
    with config.unlocked:
      config.logdir = logdir
      config.network_optimizer_d = getattr(tf.train, config.network_optimizer)
      config.network_optimizer_g = getattr(tf.train, config.network_optimizer)
      global_step = tf.Variable(1, dtype=tf.int32, name='global_step', trainable=False)

      model = config.model(config, global_step, dataset.get_input(FLAGS.celeba_path, FLAGS.batch_size, FLAGS.train), FLAGS.train, FLAGS.nb_summaries_outputs)

      sess.run(tf.global_variables_initializer())
      print("Loading Model from {}".format(FLAGS.checkpoint_used))
      model.loader.restore(sess, FLAGS.checkpoint_used)
      sess.run(tf.local_variables_initializer())

      model.generate(sess, FLAGS.batch_size, FLAGS.nb_summaries_outputs)

def main(_):
  tf_utils.set_up_logging()
  os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.GPU
  if not FLAGS.config:
    raise KeyError('You must specify a configuration.')
  if FLAGS.load_from:
    logdir = FLAGS.logdir = FLAGS.load_from
  else:
    if FLAGS.logdir and os.path.exists(FLAGS.logdir):
      run_number = [int(f.split("-")[0]) for f in os.listdir(FLAGS.logdir) if
                    os.path.isdir(os.path.join(FLAGS.logdir, f)) and FLAGS.config in f]
      run_number = max(run_number) + 1 if len(run_number) > 0 else 0
    else:
      run_number = 0
    logdir = FLAGS.logdir and os.path.expanduser(os.path.join(
      FLAGS.logdir, '{}-{}'.format(run_number, FLAGS.config)))
  try:
    config = tf_utils.load_config(logdir)
  except IOError:
    config = tools.AttrDict(getattr(configs, FLAGS.config)())
    config = tf_utils.save_config(config, logdir)
  if FLAGS.train:
    train(config, logdir)
  else:
    generate(config, logdir)

if __name__ == '__main__':
  tf.app.flags.DEFINE_string(
    'logdir', "./logdir",
    'Base directory to store logs.')
  tf.app.flags.DEFINE_string(
    'config', "std_original_tanh_gen",
    'Configuration to execute.')
  tf.app.flags.DEFINE_boolean(
    'train', True,
    'Training.')
  tf.app.flags.DEFINE_boolean(
    'resume', False,
    'Resume.')
  tf.app.flags.DEFINE_string(
    'load_from', None,
    # 'load_from', "./logdir/1-std_gan",
    'Load directory to load models from.')
  tf.app.flags.DEFINE_string(
    'GPU', "1",
    """The GPU device to run on""")
  tf.app.flags.DEFINE_string(
    'celeba_path', "../../generative-models/data/",
    """The GPU device to run on""")
  tf.app.flags.DEFINE_integer(
    'batch_size', 32,
    'batch_size')
  tf.app.flags.DEFINE_string(
    'checkpoint_used', './logdir/1-std_gan_with_tanh/models/model-12000.cptk-12000',
    'Checkpoint used to generate images'
  )
  tf.app.flags.DEFINE_integer(
    'nb_summaries_outputs', 500,
    'nb_summaries_outputs'
  )
  tf.app.run()
