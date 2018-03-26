import functools
import models
import networks

def default():
  gen_conv_layers = (4, 2, 1, 64), (4, 2, 1, 128), (4, 2, 1, 256), (4, 2, 1, 512)#, (4, 2, 0, 100)
  # gen_upsample_layers = (4, 1, 0, 512), (4, 2, 1, 256), (4, 2, 1, 128), (4, 2, 1, 64), (4, 2, 1, 3)
  gen_upsample_layers = (4, 2, 1, 256), (4, 2, 1, 128), (4, 2, 1, 64), (4, 2, 1, 3)
  disc_conv_layers = (4, 2, 1, 64), (4, 2, 1, 128), (4, 2, 1, 256), (4, 2, 1, 512), (4, 2, 0, 512)
  disc_fc_layers = 3,
  disc_middle_layer_features = 3
  nb_classes = 3
  alfa = 5e-4
  beta = 0.1 * alfa
  lr = 2e-4
  std = 0.02
  max_iters = 1e6
  summary_every = 300
  checkpoint_every = 300
  network_optimizer = 'AdamOptimizer'
  batch_norm_decay = 0.95
  weight_decay = 0.0004
  gamma = 0.5
  momentum = 0.5
  momentum2 = 0.999
  # lr_steps = [6000, 10000, 140000, 180000, 220000]
  decay_steps = 15000
  use_tanh_on_output = False
  d_iters = 5
  return locals()

def std_gan():
  locals().update(default())
  model = models.GAN
  network = networks.WNetwork
  return locals()

def std_original():
  locals().update(default())
  model = models.GAN
  network = networks.WNetwork_original
  gen_conv_layers = (5, 1, 1, 64), (4, 2, 1, 128), (4, 2, 1, 256)  # , (4, 2, 0, 100)
  gen_upsample_layers = (3, 2, 1, 128), (3, 2, 1, 64)
  gen_upsample_conv_layers = (4, 1, 1, 3),
  disc_conv_layers = (4, 2, 0, 64), (4, 2, 0, 128), (4, 2, 0, 256), (4, 2, 0, 512), (4, 2, 0, 1024)
  return locals()

def std_original_tanh_gen():
  locals().update(default())
  model = models.GAN
  network = networks.WNetwork_original
  gen_conv_layers = (5, 1, 1, 64), (4, 2, 1, 128), (4, 2, 1, 256)  # , (4, 2, 0, 100)
  gen_upsample_layers = (3, 2, 1, 128), (3, 2, 1, 64)
  gen_upsample_conv_layers = (4, 1, 1, 3),
  disc_conv_layers = (4, 2, 0, 64), (4, 2, 0, 128), (4, 2, 0, 256), (4, 2, 0, 512), (4, 2, 0, 1024)
  use_tanh_on_generator = True
  return locals()

def std_duc():
  locals().update(default())
  model = models.GAN
  network = networks.WNetwork_duc
  gen_conv_layers = (5, 1, 1, 64), (4, 2, 1, 128), (4, 2, 1, 256)  # , (4, 2, 0, 100)
  gen_upsample_layers = (3, 2, 1, 128), (3, 2, 1, 64)
  gen_upsample_conv_layers = (4, 1, 1, 3),
  disc_conv_layers = (4, 2, 0, 64), (4, 2, 0, 128), (4, 2, 0, 256), (4, 2, 0, 512), (4, 2, 0, 1024)
  return locals()

def std_gan_with_tanh():
  locals().update(default())
  model = models.GAN
  network = networks.WNetwork
  use_tanh_on_output = True
  return locals()

def std_gan_dual():
  locals().update(default())
  model = models.GAN
  network = networks.WNetwork_dual
  return locals()
