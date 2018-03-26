import tensorflow as tf
import tensorflow.contrib.layers as layers

import dataset
from tools.tf_utils import lrelu, define_saver
import os
import time
import copy

FLAGS = tf.app.flags.FLAGS

class GAN:
  def __init__(self, config, global_step, tensor_dict, is_training, nb_summaries_outputs):
    self.is_training = is_training
    self.config = config
    self.global_step = global_step
    self.init_ops = tensor_dict["init_ops"]
    self.input_1 = tensor_dict["next_element_poz"][0]
    self.label_1 = tensor_dict["next_element_poz"][1]
    self.input_0 = tensor_dict["next_element_neg"][0]
    self.label_0 = tensor_dict["next_element_neg"][1]

    self.optimizer = config.network_optimizer
    if self.is_training:
      self.model_path = os.path.join(config.logdir, "models")
      self.summary_path = os.path.join(config.logdir, "summaries")
      tf.gfile.MakeDirs(self.model_path)
      tf.gfile.MakeDirs(self.summary_path)
      self.increment_global_step = self.global_step.assign_add(1)
      self.summary_writer = tf.summary.FileWriter(self.summary_path)
      # self.summary = tf.Summary()
      self.network = config.network(config, [(self.input_1, self.label_1),
                                             (self.input_0, self.label_0)], self.is_training, self.global_step, nb_summaries_outputs)

      self.saver = self.loader = define_saver(exclude=(r'.*_temporary/.*',))
    else:
      self.summary_path = os.path.join(config.logdir, str("results_summaries_" + os.path.split(FLAGS.checkpoint_used)[1]))
      tf.gfile.MakeDirs(self.summary_path)

      self.network = config.network(config, [(self.input_1, self.label_1),
                                             (self.input_0, self.label_0)], self.is_training, self.global_step, nb_summaries_outputs)

      self.loader = define_saver(exclude=(r'.*_temporary/.*',))

  def train(self, sess):
    with sess.as_default(), sess.graph.as_default():
      # sess.run(tf.global_variables_initializer())
      sess.run(self.init_ops)
      print("Training...")
      step = sess.run(self.global_step)
      # start_time = time.time()

      while step <= self.config.max_iters:
        for _ in range(0, self.config.d_iters):
          _ = sess.run([self.network.d_train])
        _, _, summaries = sess.run([self.network.d_train, self.network.g_train, self.network.merged_summary])

        if step % self.config.summary_every == 0:
          self.summary_writer.add_summary(summaries, step)
        if step % self.config.checkpoint_every == 0:
          self.saver.save(sess, self.model_path + '/model-' + str(step) + '.cptk',
                     global_step=self.global_step)
          print("Saved Model at {}".format(self.model_path + '/model-' + str(step) + '.cptk'))

          # self.get_time_info(start_time, step)

        sess.run(self.increment_global_step)
        step += 1

  def generate(self, sess, batch_size, nb_summaries_outputs):
    with sess.as_default(), sess.graph.as_default():
      sess.run(self.init_ops)
      step = 0
      print("Generating results...")

      while step * batch_size < nb_summaries_outputs:
        eventPath = os.path.join(self.summary_path, str(step))
        tf.gfile.MakeDirs(eventPath)
        self.summary_writer = tf.summary.FileWriter(eventPath)
        _, _, summaries = sess.run([self.network.dual_res_sub, self.network.dual_res_add, self.network.merged_summary])
        self.summary_writer.add_summary(summaries, step)
        step += 1

  def get_time_info(self, start_time, current_it):
    elapsed_time = time.time() - start_time
    em, es = divmod(elapsed_time, 60)
    eh, em = divmod(em, 60)
    remaining_time = int((self.config.max_iters - current_it) / current_it * elapsed_time)
    rm, rs = divmod(remaining_time, 60)
    rh, rm = divmod(rm, 60)

    print("Elapsed time (h:m:s): {}. Remaining time (h:m:s): {}".format(
      str(int(eh)) + ':' + str(int(em)) + ':' + str(int(es)), str(int(rh)) + ':' + str(int(rm)) + ':' + str(int(rs))))
