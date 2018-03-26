import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers

from tools.tf_utils import gradient_summaries, lrelu, upsample, dense_upsample


class WNetwork():
  def __init__(self, config, input, is_training, global_step, nb_summaries_outputs):
    self.global_step = global_step
    self.is_training = is_training
    self.nb_summaries_outputs = nb_summaries_outputs
    self.config = config
    (self.input_poz, self.label_poz), (self.input_neg, self.label_neg) = input
    self.label_poz = tf.cast(self.label_poz, tf.int32)
    self.label_neg = tf.cast(self.label_neg, tf.int32)
    self.label_fake = tf.fill(tf.shape(self.label_poz), 2)
    self.image_summaries = []
    self.summaries = []

    self.learning_rate = tf.train.exponential_decay(self.config.lr, self.global_step,
                                                    self.config.decay_steps, self.config.gamma, staircase=True)
    self.network_optimizer_d = config.network_optimizer(
      self.learning_rate, name='network_optimizer_d')
    self.network_optimizer_g = config.network_optimizer(
      self.learning_rate, name='network_optimizer_g')

    self.gen_conv_layers = config.gen_conv_layers
    self.gen_upsample_layers = config.gen_upsample_layers
    self.disc_conv_layers = config.disc_conv_layers
    self.disc_middle_layer_features = config.disc_middle_layer_features

    with tf.variable_scope('generator') as scope:
      with tf.variable_scope('G1') as scope:  # positive generator
        self.res_sub = self.generator(self.input_poz, "Xpz")  # G1
        self.output_neg = self.input_poz + self.res_sub  # output_neg is a negative sample
        if self.config.use_tanh_on_output:
          self.output_neg = tf.tanh(self.output_neg)
      with tf.variable_scope('G0') as scope:  # negative generator
        self.res_add = self.generator(self.input_neg, "Xneg")  # G0
        self.output_poz = self.input_neg + self.res_add  # output_poz is a positive sample
        if self.config.use_tanh_on_output:
          self.output_poz = tf.tanh(self.output_poz)

      # ----------- Dual Ops ----------------
      with tf.variable_scope('G1') as scope:
        scope.reuse_variables()
        self.dual_res_sub = self.generator(self.output_poz, "X_hat_poz")
        self.dual_output_neg = self.output_poz + self.dual_res_sub  # dual_output_neg is a negative sample
        if self.config.use_tanh_on_output:
          self.dual_output_neg = tf.tanh(self.dual_output_neg)
      with tf.variable_scope('G0') as scope:
        scope.reuse_variables()
        self.dual_res_add = self.generator(self.output_neg, "X_hat_neg")
        self.dual_output_poz = self.output_neg + self.dual_res_add  # dual_output_poz is a positive sample
        if self.config.use_tanh_on_output:
          self.dual_output_poz = tf.tanh(self.dual_output_poz)

    with tf.variable_scope('discriminator') as scope:
      self.real_poz_logits, self.real_poz_middle_features = self.discriminator(self.input_poz)  # self.label_1
      scope.reuse_variables()
      self.real_neg_logits, self.real_neg_middle_features = self.discriminator(self.input_neg)  # self.label_0
      self.fake_neg_logits, self.fake_neg_middle_features = self.discriminator(
        self.output_neg)  # self.label_0, self.label_fake
      self.fake_poz_logits, self.fake_poz_middle_features = self.discriminator(
        self.output_poz)  # self.label_1, self.label_fake

      # ------------ Dual Ops -----------------

      self.dual_fake_poz_logits, self.dual_fake_poz_middle_features = self.discriminator(
        self.dual_output_poz)  # self.label_0, , self.label_fake
      self.dual_fake_neg_logits, self.dual_fake_neg_middle_features = self.discriminator(
        self.dual_output_neg)  # self.label_1, self.label_fake

    self.build_losses()
    self.build_optim_ops()

  def generator(self, input, name):
    out = input
    self.image_summaries.append(tf.summary.image('input_{}'.format(name), (out + 1.0) * 255 / 2.0, max_outputs=self.nb_summaries_outputs))

    with tf.variable_scope('conv'):
      for i, (kernel_size, stride, pad, nb_kernels) in enumerate(self.gen_conv_layers):
        out = layers.conv2d(out, num_outputs=nb_kernels, kernel_size=kernel_size,
                            stride=stride, activation_fn=None,
                            padding="VALID" if pad == 0 else "SAME",
                            weights_initializer=tf.random_normal_initializer(stddev=self.config.std),
                            # kernel_initializer=tf.truncated_normal_initializer(stddev=self.config.std),
                            variables_collections=tf.get_collection("generator"),
                            outputs_collections="activations", scope="conv_{}".format(i))
        out = tf.contrib.layers.batch_norm(out,
                                           center=True, scale=True, decay=self.config.batch_norm_decay,
                                           is_training=self.is_training,
                                           scope='bn_{}'.format(i))
        out = lrelu(out)
        self.summaries.append(tf.contrib.layers.summarize_activation(out))

    with tf.variable_scope("upsample"):
      for i, (kernel_size, stride, pad, nb_kernels) in enumerate(self.gen_upsample_layers):
        out = layers.conv2d_transpose(out, num_outputs=nb_kernels, kernel_size=kernel_size,
                                      stride=stride, activation_fn=None,
                                      weights_initializer=tf.random_normal_initializer(stddev=self.config.std),
                                      # kernel_initializer=tf.truncated_normal_initializer(stddev=self.config.std),
                                      padding="VALID" if pad == 0 else "SAME",
                                      variables_collections=tf.get_collection("generator"),
                                      outputs_collections="activations", scope="aux_deconv_{}".format(i))
        if i < len(self.gen_upsample_layers) - 1:
          out = tf.contrib.layers.batch_norm(out,
                                             center=True, scale=True, decay=self.config.batch_norm_decay,
                                             is_training=self.is_training,
                                             scope='bn_{}'.format(i))
          out = lrelu(out)
        # else:
        #   out = tf.tanh(out)
        self.summaries.append(tf.contrib.layers.summarize_activation(out))

    return out

  def discriminator(self, input):
    out = input
    with tf.variable_scope('conv'):
      for i, (kernel_size, stride, pad, nb_kernels) in enumerate(self.disc_conv_layers):
        out = layers.conv2d(out, num_outputs=nb_kernels, kernel_size=kernel_size,
                            stride=stride, activation_fn=None,
                            weights_initializer=tf.random_normal_initializer(stddev=self.config.std),
                            # kernel_initializer=tf.truncated_normal_initializer(stddev=self.config.std),
                            # padding="VALID" if i == len(self.disc_conv_layers) - 1 else "SAME",
                            padding="VALID" if pad == 0 else "SAME",
                            variables_collections=tf.get_collection("discriminator"),
                            outputs_collections="activations", scope="conv_{}".format(i))
        if i == self.disc_middle_layer_features:
          middle_features = out
        if i < len(self.disc_conv_layers) - 1:
          out = tf.contrib.layers.batch_norm(out,
                                             center=True, scale=True, decay=self.config.batch_norm_decay,
                                             is_training=self.is_training,
                                             scope='bn_{}'.format(i))
          out = lrelu(out)
        else:
          # out = tf.nn.softmax(layers.flatten(out))
          out = layers.flatten(out)
        self.summaries.append(tf.contrib.layers.summarize_activation(out))

      with tf.variable_scope("fc"):
        for i, nb_filt in enumerate(self.config.disc_fc_layers):
          out = layers.fully_connected(out, num_outputs=nb_filt,
                                       activation_fn=None,
                                       variables_collections=tf.get_collection("variables"),
                                       outputs_collections="activations", scope="fc_{}".format(i))
          self.summaries.append(tf.contrib.layers.summarize_activation(out))

      return out, middle_features

  def build_losses(self):
    # ------------------------------------ D ---------------------------------------
    self.loss_cls_input_poz = tf.nn.softmax_cross_entropy_with_logits(
      logits=self.real_poz_logits,
      labels=tf.one_hot(self.label_poz, self.config.nb_classes))
    self.loss_cls_input_neg = tf.nn.softmax_cross_entropy_with_logits(
      logits=self.real_neg_logits,
      labels=tf.one_hot(self.label_neg, self.config.nb_classes))
    self.loss_cls_output_poz = tf.nn.softmax_cross_entropy_with_logits(
      logits=self.fake_poz_logits,
      labels=tf.one_hot(self.label_fake, self.config.nb_classes))
    self.loss_cls_output_neg = tf.nn.softmax_cross_entropy_with_logits(
      logits=self.fake_neg_logits,
      labels=tf.one_hot(self.label_fake, self.config.nb_classes))
    self.d_vars = tf.get_collection("variables", "discriminator")
    #self.l2_losses_d = tf.add_n([self.config.weight_decay * tf.nn.l2_loss(v) for v in self.d_vars])
    self.d_loss = tf.reduce_mean(self.loss_cls_input_poz + self.loss_cls_input_neg \
                                 + self.loss_cls_output_poz + self.loss_cls_output_neg)
    #self.d_loss_total = self.d_loss + self.l2_losses_d
    self.d_loss_total = self.d_loss

    # ------------------------------------ G0 ---------------------------------------

    self.g0_gan = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
      logits=self.fake_poz_logits,
      labels=tf.one_hot(self.label_poz, self.config.nb_classes)))
    self.g0_pix = tf.reduce_mean(tf.reduce_sum(tf.abs(self.res_add), axis=[1, 2, 3]))
    self.g0_per = tf.reduce_mean(tf.reduce_sum(
      tf.abs(self.real_neg_middle_features - self.fake_poz_middle_features), axis=[1, 2, 3]))
    self.g0_dual = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
      logits=self.dual_fake_poz_logits,
      labels=tf.one_hot(self.label_poz, self.config.nb_classes)))
    self.g0_loss = self.g0_gan + self.g0_dual + self.config.alfa * self.g0_pix + self.config.beta * self.g0_per

    # ------------------------------------ G1 ---------------------------------------

    self.g1_gan = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
      logits=self.fake_neg_logits,
      labels=tf.one_hot(self.label_neg, self.config.nb_classes)))
    self.g1_pix = tf.reduce_mean(tf.reduce_sum(tf.abs(self.res_sub), axis=[1, 2, 3]))
    self.g1_per = tf.reduce_mean(tf.reduce_sum(
      tf.abs(self.real_poz_middle_features - self.fake_neg_middle_features), axis=[1, 2, 3]))
    self.g1_dual = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
      logits=self.dual_fake_neg_logits,
      labels=tf.one_hot(self.label_neg, self.config.nb_classes)))
    self.g1_loss = self.g1_gan + self.g1_dual + self.config.alfa * self.g1_pix + self.config.beta * self.g1_per
    self.g_vars = tf.get_collection("variables", "generator")
    #self.l2_losses_g = tf.add_n([self.config.weight_decay * tf.nn.l2_loss(v) for v in self.g_vars])
    self.g_loss = self.g0_loss + self.g1_loss
    #self.g_loss_total = self.g_loss + self.l2_losses_g
    self.g_loss_total = self.g_loss

  def build_optim_ops(self):
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      d_gradients = self.network_optimizer_d.compute_gradients(self.d_loss_total, var_list=self.d_vars)
      d_clipped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) if grad is not None else (grad, var) for grad, var
                             in
                             d_gradients]
      self.d_train = self.network_optimizer_d.apply_gradients(d_clipped_gradients)

      g_gradients = self.network_optimizer_g.compute_gradients(self.g_loss_total, var_list=self.g_vars)
      g_clipped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) if grad is not None else (grad, var) for grad, var
                             in
                             g_gradients]
      self.g_train = self.network_optimizer_g.apply_gradients(g_clipped_gradients)

    self.loss_summaries = [tf.summary.scalar('d_loss', self.d_loss),
                           tf.summary.scalar('d_loss_total', self.d_loss_total),
                           #tf.summary.scalar('l2_losses_d', self.l2_losses_d),
                           tf.summary.scalar('g_loss', self.g_loss),
                           tf.summary.scalar('g_loss_total', self.g_loss_total),
                           #tf.summary.scalar('l2_losses_g', self.l2_losses_g),
                           tf.summary.scalar('g0_loss', self.g0_loss),
                           tf.summary.scalar('g0_gan', self.g0_gan),
                           tf.summary.scalar('g0_dual', self.g0_dual),
                           tf.summary.scalar('g0_pix', self.g0_pix),
                           tf.summary.scalar('g0_per', self.g0_per),
                           tf.summary.scalar('g1_loss', self.g1_loss),
                           tf.summary.scalar('g1_gan', self.g1_gan),
                           tf.summary.scalar('g1_dual', self.g1_dual),
                           tf.summary.scalar('g1_pix', self.g1_pix),
                           tf.summary.scalar('g1_per', self.g1_per),
                           tf.summary.scalar("lr", self.learning_rate)]

    self.merged_summary = tf.summary.merge(self.image_summaries + self.summaries + self.loss_summaries +
                                           [gradient_summaries(g_gradients) + gradient_summaries(d_gradients)])


class WNetwork_original(WNetwork):
  def generator(self, input, name):
    out = input
    self.image_summaries.append(tf.summary.image('input_{}'.format(name), (out + 1.0) * 255 / 2.0, max_outputs=self.nb_summaries_outputs))

    with tf.variable_scope('conv'):
      for i, (kernel_size, stride, pad, nb_kernels) in enumerate(self.gen_conv_layers):
        out = layers.conv2d(out, num_outputs=nb_kernels, kernel_size=kernel_size,
                            stride=stride, activation_fn=None,
                            padding="VALID" if pad == 0 else "SAME",
                            weights_initializer=tf.random_normal_initializer(stddev=self.config.std),
                            # kernel_initializer=tf.truncated_normal_initializer(stddev=self.config.std),
                            variables_collections=tf.get_collection("generator"),
                            outputs_collections="activations", scope="conv_{}".format(i))
        out = tf.contrib.layers.batch_norm(out,
                                           center=True, scale=True, decay=self.config.batch_norm_decay,
                                           is_training=self.is_training,
                                           scope='bn_{}'.format(i))
        out = lrelu(out)
        self.summaries.append(tf.contrib.layers.summarize_activation(out))

    with tf.variable_scope("deconv"):
      for i, (kernel_size, stride, pad, nb_kernels) in enumerate(self.gen_upsample_layers):
        out = upsample(out, nb_kernels=nb_kernels, kernel_size=kernel_size, pad=pad, is_training=self.is_training,
                       batch_norm_decay=self.config.batch_norm_decay,
                       weights_initializer=tf.random_normal_initializer(stddev=self.config.std), stride=stride,
                       variables_collection="generator", nblayer=i)

        self.summaries.append(tf.contrib.layers.summarize_activation(out))

    with tf.variable_scope("deconv_conv"):
      for i, (kernel_size, stride, pad, nb_kernels) in enumerate(self.config.gen_upsample_conv_layers):
        out = layers.conv2d(out, num_outputs=nb_kernels, kernel_size=kernel_size,
                            stride=stride, activation_fn=None,
                            padding="VALID" if pad == 0 else "SAME",
                            weights_initializer=tf.random_normal_initializer(stddev=self.config.std),
                            # kernel_initializer=tf.truncated_normal_initializer(stddev=self.config.std),
                            variables_collections=tf.get_collection("generator"),
                            outputs_collections="activations", scope="deconv_conv_{}".format(i))
        if self.config.use_tanh_on_generator:
          out = tf.tanh(out)
    return out


class WNetwork_duc(WNetwork):
  def generator(self, input, name):
    out = input
    self.image_summaries.append(tf.summary.image('input_{}'.format(name), (out + 1.0) * 255 / 2.0, max_outputs=self.nb_summaries_outputs))

    with tf.variable_scope('conv'):
      for i, (kernel_size, stride, pad, nb_kernels) in enumerate(self.gen_conv_layers):
        out = layers.conv2d(out, num_outputs=nb_kernels, kernel_size=kernel_size,
                            stride=stride, activation_fn=None,
                            padding="VALID" if pad == 0 else "SAME",
                            weights_initializer=tf.random_normal_initializer(stddev=self.config.std),
                            # kernel_initializer=tf.truncated_normal_initializer(stddev=self.config.std),
                            variables_collections=tf.get_collection("generator"),
                            outputs_collections="activations", scope="conv_{}".format(i))
        out = tf.contrib.layers.batch_norm(out,
                                           center=True, scale=True, decay=self.config.batch_norm_decay,
                                           is_training=self.is_training,
                                           scope='bn_{}'.format(i))
        out = lrelu(out)
        self.summaries.append(tf.contrib.layers.summarize_activation(out))

    with tf.variable_scope("deconv"):
      for i, (kernel_size, stride, pad, nb_kernels) in enumerate(self.gen_upsample_layers):
        out = dense_upsample(out, nb_kernels=nb_kernels, kernel_size=kernel_size, pad=pad, is_training=self.is_training,
                             batch_norm_decay=self.config.batch_norm_decay,
                             weights_initializer=tf.random_normal_initializer(stddev=self.config.std), stride=stride,
                             variables_collection="generator", nblayer=i)
        self.summaries.append(tf.contrib.layers.summarize_activation(out))

    with tf.variable_scope("deconv_conv"):
      for i, (kernel_size, stride, pad, nb_kernels) in enumerate(self.config.gen_upsample_conv_layers):
        out = layers.conv2d(out, num_outputs=nb_kernels, kernel_size=kernel_size,
                            stride=stride, activation_fn=None,
                            padding="VALID" if pad == 0 else "SAME",
                            weights_initializer=tf.random_normal_initializer(stddev=self.config.std),
                            # kernel_initializer=tf.truncated_normal_initializer(stddev=self.config.std),
                            variables_collections=tf.get_collection("generator"),
                            outputs_collections="activations", scope="deconv_conv_{}".format(i))
    return out


class WNetwork_dual(WNetwork):
  def __init__(self, config, input, is_training):
    self.is_training = is_training
    self.config = config
    self.input_1, self.input_0 = input
    self.image_summaries = []
    self.summaries = []

    self.network_optimizer = config.network_optimizer(
      self.config.lr, name='network_optimizer')

    self.gen_conv_layers = config.gen_conv_layers
    self.gen_upsample_layers = config.gen_upsample_layers
    self.disc_conv_layers = config.disc_conv_layers
    self.disc_middle_layer_features = config.disc_middle_layer_features

    with tf.variable_scope('generator_poz') as scope:
      self.res_1 = self.generator(self.input_1, "X0")  # G1
      self.output_1 = self.input_1 + self.res_1
      # scope.reuse_variables()
    with tf.variable_scope('generator_neg') as scope:
      self.res_0 = self.generator(self.input_0, "X1")  # G0
      self.output_0 = self.input_0 + self.res_0

      # ----------- Dual Ops ----------------
    with tf.variable_scope('generator_neg') as scope:
      scope.reuse_variables()
      self.dual_res_0 = self.generator(self.output_0, "X_hat_0")
      self.dual_output_0 = self.output_0 + self.dual_res_0

    with tf.variable_scope('generator_poz') as scope:
      scope.reuse_variables()
      self.dual_res_1 = self.generator(self.output_1, "X_hat_1")
      self.dual_output_1 = self.output_1 + self.dual_res_1

    with tf.variable_scope('discriminator') as scope:
      self.real_1_logits, real_1_middle_features = self.discriminator(self.input_1)
      scope.reuse_variables()
      self.real_0_logits, real_0_middle_features = self.discriminator(self.input_0)
      self.fake_1_logits, fake_1_middle_features = self.discriminator(self.output_1)
      self.fake_0_logits, fake_0_middle_features = self.discriminator(self.output_0)

      # ------------ Dual Ops -----------------

      self.dual_fake_0_logits, dual_fake_0_middle_features = self.discriminator(self.dual_output_0)
      self.dual_fake_1_logits, dual_fake_1_middle_features = self.discriminator(self.dual_output_1)

    self.build_losses()
    self.build_optim_ops()
