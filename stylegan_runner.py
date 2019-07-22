import os
import pdb
import sys
import pickle
import random
import math
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm_notebook as tqdm

def easygen_train(model_path, images_path, dataset_path, start_kimg=7000, max_kimg=25000, schedule='', seed=1000):
  #import stylegan
  #from stylegan import config
  ##from stylegan import dnnlib
  #from stylegan.dnnlib import EasyDict

  #images_dir = '/content/raw'
  #max_kimg = 25000
  #start_kimg = 7000
  #schedule = ''
  #model_in = '/content/karras2019stylegan-cats-256x256.pkl'

  #dataset_dir = '/content/stylegan_dataset' #os.path.join(cwd, 'cache', 'stylegan_dataset')
  
  import config
  config.data_dir = '/content/datasets'
  config.results_dir = '/content/results'
  config.cache_dir = '/contents/cache'
  run_dir_ignore = ['/contents/results', '/contents/datasets', 'contents/cache']
  import copy
  import dnnlib
  from dnnlib import EasyDict
  from metrics import metric_base
  # Prep dataset
  import dataset_tool
  print("prepping dataset...")
  dataset_tool.create_from_images(tfrecord_dir=dataset_path, image_dir=images_path, shuffle=False)
  # Set up training parameters
  desc          = 'sgan'                                                                 # Description string included in result subdir name.
  train         = EasyDict(run_func_name='training.training_loop.training_loop')         # Options for training loop.
  G             = EasyDict(func_name='training.networks_stylegan.G_style')               # Options for generator network.
  D             = EasyDict(func_name='training.networks_stylegan.D_basic')               # Options for discriminator network.
  G_opt         = EasyDict(beta1=0.0, beta2=0.99, epsilon=1e-8)                          # Options for generator optimizer.
  D_opt         = EasyDict(beta1=0.0, beta2=0.99, epsilon=1e-8)                          # Options for discriminator optimizer.
  G_loss        = EasyDict(func_name='training.loss.G_logistic_nonsaturating')           # Options for generator loss.
  D_loss        = EasyDict(func_name='training.loss.D_logistic_simplegp', r1_gamma=10.0) # Options for discriminator loss.
  dataset       = EasyDict()                                                             # Options for load_dataset().
  sched         = EasyDict()                                                             # Options for TrainingSchedule.
  grid          = EasyDict(size='1080p', layout='random')                                # Options for setup_snapshot_image_grid().
  #metrics       = [metric_base.fid50k]                                                  # Options for MetricGroup.
  submit_config = dnnlib.SubmitConfig()                                                  # Options for dnnlib.submit_run().
  tf_config     = {'rnd.np_random_seed': seed}                                           # Options for tflib.init_tf().
  # Dataset
  desc                            += '-custom'
  dataset                         = EasyDict(tfrecord_dir=dataset_path)
  train.mirror_augment            = False
  # Number of GPUs.
  desc                            += '-1gpu'
  submit_config.num_gpus          = 1
  sched.minibatch_base            = 4
  sched.minibatch_dict            = {4: 128, 8: 128, 16: 128, 32: 64, 64: 32, 128: 16, 256: 8, 512: 4} #{4: 256, 8: 256, 16: 128, 32: 64, 64: 32, 128: 16, 256: 16}
  # Default options.
  train.total_kimg                = max_kimg
  sched.lod_initial_resolution    = 8
  sched.G_lrate_dict              = {128: 0.0015, 256: 0.002, 512: 0.003, 1024: 0.003}
  sched.D_lrate_dict              = EasyDict(sched.G_lrate_dict)
  # schedule
  schedule_dict = {4: 160, 8:140, 16:120, 32:100, 64:80, 128:60, 256:40, 512:30, 1024:20} #{4: 2, 8:2, 16:2, 32:2, 64:2, 128:2, 256:2, 512:2, 1024:2} # Runs faster for small datasets
  if len(schedule) >=5 and schedule[0] == '{' and schedule[-1] == '}' and ':' in schedule:
    # is schedule a string of a dict?
    try:
      temp = eval(schedule)
      schedule_dict = dict(temp)
      # assert: it is a dict
    except:
      pass
  elif len(schedule) > 0:    
    # is schedule an int?
    try:
      schedule_int = int(schedule)
      #assert: schedule is an int
      schedule_dict = {}
      for i in range(1, 10):
        schedule_dict[int(math.pow(2, i+1))] = schedule_int
    except:
      pass      
  print('schedule:', str(schedule_dict))
  sched.tick_kimg_dict = schedule_dict
  # resume kimg
  resume_kimg                     = start_kimg
  # path to model
  resume_run_id                   = model_path
  # tick snapshots
  image_snapshot_ticks            = 1
  network_snapshot_ticks          = 1
  # Submit run
  kwargs = EasyDict(train)
  kwargs.update(G_args=G, D_args=D, G_opt_args=G_opt, D_opt_args=D_opt, G_loss_args=G_loss, D_loss_args=D_loss)
  kwargs.update(dataset_args=dataset, sched_args=sched, grid_args=grid, tf_config=tf_config)
  kwargs.update(resume_kimg=resume_kimg, resume_run_id=resume_run_id)
  kwargs.update(image_snapshot_ticks=image_snapshot_ticks, network_snapshot_ticks=network_snapshot_ticks)
  kwargs.submit_config = copy.deepcopy(submit_config)
  kwargs.submit_config.run_dir_root = dnnlib.submission.submit.get_template_from_path(config.result_dir)
  kwargs.submit_config.run_dir_ignore += config.run_dir_ignore
  kwargs.submit_config.run_desc = desc
  dnnlib.submit_run(**kwargs)
  
def easygen_run(model_path, images_path, num=1):
  # from https://github.com/ak9250/stylegan-art/blob/master/styleganportraits.ipynb
  truncation = 0.7 # hard coding because everyone uses this value
  import dnnlib
  import dnnlib.tflib as tflib
  import config
  tflib.init_tf()
  #num = 10
  #model = '/content/karras2019stylegan-cats-256x256.pkl'
  #images_dir = '/content/cache/run_out'
  #truncation = 0.7
  _G = None
  _D = None
  Gs = None
  with open(model_path, 'rb') as f:
      _G, _D, Gs = pickle.load(f)
  fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
  synthesis_kwargs = dict(output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True), minibatch_size=8)
  latents = np.random.RandomState(int(1000*random.random())).randn(num, *Gs.input_shapes[0][1:])
  labels = np.zeros([latents.shape[0]] + Gs.input_shapes[1][1:])
  images = Gs.run(latents, None, truncation_psi=truncation, randomize_noise=False, output_transform=fmt)
  for n, image in enumerate(images):
    # img = Image.fromarray(images[0])
    img = Image.fromarray(image)
    img.save(os.path.join(images_path, str(n) + '.jpg'), "JPEG")

def get_latent_interpolation(endpoints, num_frames_per, mode = 'linear', shuffle = False):
    if shuffle:
        random.shuffle(endpoints)
    num_endpoints, dim = len(endpoints), len(endpoints[0])
    num_frames = num_frames_per * num_endpoints
    endpoints = np.array(endpoints)
    latents = np.zeros((num_frames, dim))
    for e in range(num_endpoints):
        e1, e2 = e, (e+1)%num_endpoints
        for t in range(num_frames_per):
            frame = e * num_frames_per + t
            r = 0.5 - 0.5 * np.cos(np.pi*t/(num_frames_per-1)) if mode == 'ease' else float(t) / num_frames_per
            latents[frame, :] = (1.0-r) * endpoints[e1,:] + r * endpoints[e2,:]
    return latents


    
def easygen_movie(model_path, movie_path, num=10, interp=10, duration=10):
  # from https://github.com/ak9250/stylegan-art/blob/master/styleganportraits.ipynb
  import dnnlib
  import dnnlib.tflib as tflib
  import config
  tflib.init_tf()
  truncation = 0.7 # what everyone uses
  # Get model
  _G = None
  _D = None
  Gs = None
  with open(model_path, 'rb') as f:
    _G, _D, Gs = pickle.load(f)
  # Make waypoints
  #fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
  #synthesis_kwargs = dict(output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True), minibatch_size=8)
  waypoint_latents = np.random.RandomState(int(1000*random.random())).randn(num, *Gs.input_shapes[0][1:])
  #waypoint_labels = np.zeros([waypoint_latents.shape[0]] + Gs.input_shapes[1][1:])
  #waypoint_images = Gs.run(latents, None, truncation_psi=truncation, randomize_noise=False, output_transform=fmt)
  # interpolate
  interp_latents = get_latent_interpolation(waypoint_latents, interp)
  interp_labels = np.zeros([interp_latents.shape[0]] + Gs.input_shapes[1][1:])
  fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
  synthesis_kwargs = dict(output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True), minibatch_size=8)
  batch_size = 8
  num_frames = interp_latents.shape[0]
  num_batches = int(np.ceil(num_frames/batch_size))
  images = []
  for b in tqdm(range(num_batches)):
      new_images = Gs.run(interp_latents[b*batch_size:min((b+1)*batch_size, num_frames-1), :], None, truncation_psi=truncation, randomize_noise=False, output_transform=fmt)
      for img in new_images:
          images.append(Image.fromarray(img)) # convert to PIL.Image
  images[0].save(movie_path, "GIF",
                 save_all=True,
                 append_images=images[1:],
                 duration=duration,
                 loop=0)

    
if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Process runner commands.')
  parser.add_argument('--train', action="store_true", default=False)
  parser.add_argument('--run', action="store_true", default=False)
  parser.add_argument('--movie', action="store_true", default=False)
  parser.add_argument("--model", help="model to load", default="")
  parser.add_argument("--images_in", help="directory containing training images", default="")
  parser.add_argument("--images_out", help="diretory to store generated images", default="")
  parser.add_argument("--movie_out", help="directory to save movie", default="")
  parser.add_argument("--dataset_temp", help="where to store prepared image data", default="")
  parser.add_argument("--schedule", help="training schedule", default="")
  parser.add_argument("--max_kimg", help="iteration to stop training at", type=int, default=25000)
  parser.add_argument("--start_kimg", help="iteration to start training at", type=int, default=7000)
  parser.add_argument("--num", help="number of images to generate", type=int, default=1)
  parser.add_argument("--interp", help="number of images to interpolate", type=int, default=10)
  parser.add_argument("--duration", help="how long for each image in movie", type=int, default=10)
  parser.add_argument("--seed", help="seed number", type=int, default=1000)
  args = parser.parse_args()
  if args.train:
    easygen_train(model_path=args.model, 
                  images_path=args.images_in,
                  dataset_path=args.dataset_temp,
                  start_kimg=args.start_kimg,
                  max_kimg=args.max_kimg,
                  schedule=args.schedule,
                  seed=args.seed)
  elif args.run:
    easygen_run(model_path=args.model,
                images_path=args.images_out,
                num=args.num)
  elif args.movie:
    easygen_movie(model_path=args.model,
                  movie_path=args.movie_out,
                  num=args.num,
                  interp=args.interp,
                  duration=args.duration)
