import os
import scipy.misc
import numpy as np

from model import DCGAN
from utils import pp, visualize, to_json, show_all_variables

import tensorflow as tf
'''
flags = tf.app.flags
flags.DEFINE_integer("epoch", 25, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("input_height", 108, "The size of image to use (will be center cropped). [108]")
flags.DEFINE_integer("input_width", None, "The size of image to use (will be center cropped). If None, same value as input_height [None]")
flags.DEFINE_integer("output_height", 64, "The size of the output images to produce [64]")
flags.DEFINE_integer("output_width", None, "The size of the output images to produce. If None, same value as output_height [None]")
flags.DEFINE_string("dataset", "celebA", "The name of dataset [celebA, mnist, lsun]")
flags.DEFINE_string("input_fname_pattern", "*.jpg", "Glob pattern of filename of input images [*]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("train", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("crop", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")
FLAGS = flags.FLAGS
'''

def run(epoch = 25, learning_rate = 0.0002, beta1 = 0.5, train_size = np.inf, batch_size = 64, input_height = 108, input_width = None, output_height = 64, output_width = None, dataset = 'celebA', input_fname_pattern = '*.jpg', checkpoint_dir = 'checkpoints', sample_dir = 'samples', output_dir = 'output', train = False, crop = False, num_images = 1):
  #pp.pprint(flags.FLAGS.__flags)

  if input_width is None:
    input_width = input_height
  if output_width is None:
    output_width = output_height

  #if not os.path.exists(checkpoint_dir):
  #  os.makedirs(checkpoint_dir)
  if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
  run_config = tf.ConfigProto()
  run_config.gpu_options.allow_growth=True

  with tf.Session(config=run_config) as sess:
    dcgan = DCGAN(
        sess,
        input_width=input_width,
        input_height=input_height,
        output_width=output_width,
        output_height=output_height,
        batch_size=batch_size,
        sample_num=batch_size,
        dataset_name=dataset,
        input_fname_pattern=input_fname_pattern,
        crop=crop,
        checkpoint_dir=checkpoint_dir,
        sample_dir=sample_dir,
        output_dir=output_dir)

    show_all_variables()

    if train:
      dcgan.train(epoch = epoch, learning_rate = learning_rate, beta1 = beta1, train_size = train_size, batch_size = batch_size, input_height = input_height, input_width = input_width, output_height = output_height, output_width = output_width, dataset = dataset, input_fname_pattern = input_fname_pattern, checkpoint_dir = checkpoint_dir, sample_dir = sample_dir, output_dir = output_dir, train = train, crop = crop)
    #else:
    #  if not dcgan.load(checkpoint_dir)[0]:
    #    raise Exception("[!] Train a model first, then run test mode")
    else:  
      # Below is codes for visualization
      OPTION = 0
      for n in range(num_images):
        visualize(sess, dcgan, option = OPTION, epoch = epoch, learning_rate = learning_rate, beta1 = beta1, train_size = train_size, batch_size = batch_size, input_height = input_height, input_width = input_width, output_height = output_height, output_width = output_width, dataset = dataset, input_fname_pattern = input_fname_pattern, checkpoint_dir = checkpoint_dir, sample_dir = sample_dir, output_dir = output_dir, train = train, crop = crop, count = n)

