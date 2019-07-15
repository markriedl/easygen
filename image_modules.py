from module import *
import requests
import os
import re
import time
import random
import pickle
import numpy as np
import math
import copy
from torchvision import transforms
from torchvision import utils
from PIL import Image
import shutil
import pdb
import subprocess
import sys

#aaah



##############################

class ScrapePinterest(Module):

    def __init__(self, url, username, password, target, output):
        self.url = url              # Initial url (must be at Pinterest)
        self.username = username    # Pinterest username
        self.password = password    # Pinterest password
        self.target = target        # target number of images to download
        self.output = output        # path to output directory to store image files
        self.ready = True
        self.output_files = [output]

    def run(self):
        # Do some input checking
        if 'pinterest.com' not in self.url.lower():
            print("url (" + self.url + ") is not a pinterest.com url.")
            return
        if len(self.password) == 0:
            print("password is empty")
            return
        if len(self.username) == 0:
            print("username is empty")
            return
        from selenium import webdriver
        from selenium.webdriver.common.keys import Keys
        # Set up Chrome Driver
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        wd = webdriver.Chrome('chromedriver',options=options)
        # open Pinterest and log in
        print("logging in to Pinterest...")
        wd.get("https://www.pinterest.com")
        emailElem = wd.find_element_by_name('id')
        emailElem.send_keys(self.username)
        passwordElem = wd.find_element_by_name('password')
        passwordElem.send_keys()
        passwordElem.send_keys(Keys.RETURN)
        time.sleep(5 + random.randint(1, 5))
        # Get the first page
        print("Going to first page...")
        wd.get(self.url)
        # sleep
        time.sleep(5 + random.randint(1, 5))
        # get urls for images
        results = set() # the urls
        url_count = 0   # how many image urls have we found?
        miss_count = 0  # how many times have we failed to get more images?
        max_miss_count = 10 # how many times are we willing to try again?
        print("getting image urls...")
        while len(results) < self.target and miss_count < 5:
            # Find image elements in the web page
            images = wd.find_elements_by_tag_name("img")
            # Iterate through the elements and try to get the urls, which is the source of the element
            for i in images:
                try:
                    src = i.get_attribute("src")
                    results.add(src)
                except:
                    pass
            # Did we fail to get new image urls?
            print(len(results))
            if len(results) == url_count:
                # If so, increment miss count
                miss_count = miss_count + 1
            # Remember how many image urls we had last time around
            url_count = len(results)
            # sleep
            time.sleep(5 + random.randint(1, 5))
            # Send page down signal
            dummy = wd.find_element_by_tag_name('a')
            dummy.send_keys(Keys.PAGE_DOWN)
        # convert results to list
        results = list(results)
        # Prep download directory
        prep_output_dir(self.output)
        # download images
        print("downloading images...")
        for result in results:
            res = requests.get(result)
            filename = os.path.join(self.output, os.path.basename(result))
            file = open(filename, 'wb')
            for chunk in res.iter_content(100000):
                file.write(chunk)
            file.close()
            

####################################

class ResizeImages(Module):

    def __init__(self, input, size, output):
        self.input = input          # path to input files (directory)
        self.output = output        # path to output files (directory)
        self.size = size            # width and height (int)
        self.ready = checkFiles(input)
        self.output_files = [output]

    def run(self):
        # Prep output file directory
        prep_output_dir(self.output)
        # The transformation
        p = transforms.Compose([transforms.Resize((self.size,self.size))])
        # Apply to all files in input directory
        if os.path.exists(self.input) and os.path.isdir(self.input):
            for file in os.listdir(self.input):
                try:
                    img = Image.open(os.path.join(self.input, file))
                    img2 = p(img)
                    img2.save(os.path.join(self.output, file), 'JPEG')
                except:
                    print(file, "did not load")
        else:
            print(self.input, "is not a directory")

###################################

### Module for face detection/close cropping faces

class CropFaces(Module):

    def __init__(self, input, size, output, rejects):
        self.input = input          # path to directory containing images
        self.output = output        # path to directory to save new images
        self.rejects = rejects      # path to directory to save rejected images
        self.size = size            # (int) size of height and width of output files
        self.ready = checkFiles(input)
        self.output_files = [output, rejects]

    def run(self):
        # Prep output file directory
        prep_output_dir(self.output)
        prep_output_dir(self.rejects)
        # Run autocrop program
        cmd = 'autocrop -i ' + self.input + ' -o ' + self.output + ' -w ' + str(self.size) + ' -H ' + str(self.size) + ' --facePercent 50 -r ' + self.rejects
        status = os.system(cmd)

### Module for removing grayscale images
class RemoveGrayscale(Module):

    def __init__(self, input, output, rejects):
        self.input = input          # path to directory containing images
        self.output = output        # path to directory to put non-grayscale images
        self.rejects = rejects      # path to directory to put grayscale images
        self.ready = checkFiles(input)
        self.output_files = [output, rejects]

    def run(self):
        # prep output file directory
        prep_output_dir(self.output, makedir=True)
        prep_output_dir(self.rejects, makedir=True)
        # Check each file
        for file in os.listdir(self.input):
            try:
                img = Image.open(os.path.join(self.input, file))
            except:
                print(file, "could not be loaded")
            if len(img.getbands()) == 3:
                # this images has 3 color channels (not grayscale)
                shutil.copyfile(os.path.join(self.input, file), os.path.join(self.output, file))
            else:
                # this image is grayscale
                shutil.copyfile(os.path.join(self.input, file), os.path.join(self.rejects, file))

#####################

class MakeGrayscale(Module):

    def __init__(self, input, output):
        self.input = input          # path to directory containing images
        self.output = output        # path to directory
        self.ready = checkFiles(input)
        self.output_files = [output]

    def run(self):
        # prep output file directory
        prep_output_dir(self.output)
        # load each file
        for file in os.listdir(self.input):
            # load and convert
            img = Image.open(os.path.join(self.input, file)).convert('LA')
            # save to target destination
            img.save(os.path.join(self.output, file))

#############################

### Module for saving images (directory)
class SaveImages(Module):

    def __init__(self, images, directory):
        self.images = images        # path to directory containing images
        self.directory = directory  # path name to save to
        self.ready = checkFiles(images)
        self.output_files = [directory]

    def run(self):
        prep_output_dir(self.directory, makedir=False)
        shutil.copytree(self.images, self.directory)


###############################

class LoadImages(Module):

    def __init__(self, directory, images):
        self.images = images            # path to save images into
        self.directory = directory       # path to load images from
        self.ready = True
        self.output_files = [images]

    def run(self):
        # Check for path existence
        if os.path.exists(self.directory):
            # If it's a directory, copy all files in the directory
            if os.path.isdir(self.directory):
                prep_output_dir(self.images, makedir=False)
                shutil.copytree(self.directory, self.images)
            else:
                # Not a directory, make a directory and copy the single file into it
                prep_output_dir(self.images)
                shutil.copy(self.directory, self.images)

################################

### Module for fine-tuning StyleGAN
### (include output for an animation)

class StyleGAN_FineTune(Module):

    def __init__(self, model_in, images, start_kimg, max_kimg, seed, schedule, model_out, animation):
        self.model_in = model_in                # path to model (pkl file)
        self.images = images                    # path to image directory
        self.start_kimg = start_kimg            # (int) iteration to begin with (closer to zero means retrain more) default: 7000
        self.max_kimg = max_kimg                # (int) max kimg
        self.schedule = schedule                # (int or string or dict) the number of kimgs per resolution level
        self.model_out = model_out              # path to fine tuned model (pkl file)
        self.animation = animation              # path to an animation file
        self.seed = seed                        # (int) default = 1000
        self.ready = checkFiles(model_in, images)
        self.output_files = [model_out, animation]

    def run(self):
        print("Keyboard interrupt will stop training but program will try to continue.")
        # Run the stylegan program in a separate sub-process
        cwd = os.getcwd()
        params = {'model': os.path.join(cwd, self.model_in),
                  'images_in' : os.path.join(cwd, self.images),
                  'dataset_temp' : os.path.join(cwd, 'stylegan_dataset'),
                  'start_kimg' : self.start_kimg,
                  'max_kimg' : self.max_kimg,
                  'schedule' : self.schedule,
                  'seed' : self.seed
                 }
        command = 'python stylegan/stylegan_runner.py --train'
        for key in params.keys():
          val = params[key]
          command = command + ' --' + str(key) + ' ' + str(val)
        print("launching", command)
        try:
          process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True )
          for line in iter(process.stdout.readline, b''):  # replace '' with b'' for Python 3
            sys.stdout.write(line)
        except KeyboardInterrupt:
          print("Keyboard interrupt")
        ### ASSERT: we are done training
        # Get final model... it's the pkl with the biggest number
        # First, get the latest run directory
        run_dir = ''
        latest = -1
        for file in os.listdir(os.path.join(cwd, 'results')):
            match = re.match(r'([0-9]+)', file)
            if match is not None and match.group(1) is not None:
                current = int(match.group(1))
                if current > latest:
                    latest = current
                    run_dir = file
        run_dir = os.path.join(cwd, 'results', run_dir)
        # Now get the newest pkl file and image files
        model_filename = ''
        image_files = []
        latest = 0.0
        for file in os.listdir(run_dir):
            match_png = re.search(r'[\w\W]*?[0-9]+?.png', file)
            match_pkl = re.search(r'[\w\W]*?[0-9]+?.pkl', file)
            if match_pkl is not None:
                current = os.stat(os.path.join(cwd, 'results', run_dir, file)).st_ctime
                if current > latest:
                    latest = current
                    model_filename = file
            elif match_png is not None:
                image_files.append(os.path.join(cwd, 'results', run_dir, file))
        model_filename = os.path.join(cwd, 'results', run_dir, model_filename)
        # save animation images
        prep_output_dir(self.animation)
        for file in image_files:
          shutil.copy(file, self.animation)
        # Save fine tuned model
        shutil.copyfile(model_filename, self.model_out)



############################



class StyleGAN_Run(Module):

    def __init__(self, model, num, images):
        self.model = model              # path to model
        self.num = num                  # number of images to generate (int)
        self.images = images              # path to output image
        self.ready = checkFiles(model)
        self.output_files = [images]

    def run(self):
        prep_output_dir(self.images)
        # Run the stylegan program in a separate sub-process
        cwd = os.getcwd()
        params = {'model': os.path.join(cwd, self.model),
                  'images_out' : os.path.join(cwd, self.images),
                  'num' : self.num,
                 }
        command = 'python stylegan/stylegan_runner.py --run'
        for key in params.keys():
          val = params[key]
          command = command + ' --' + str(key) + ' ' + str(val)
        print("launching", command)
        try:
          process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True )
          for line in iter(process.stdout.readline, b''):  # replace '' with b'' for Python 3
            sys.stdout.write(line)
        except Exception as e:
          print("something broke")
          print(e)


#############

# Not tested. Needs to launch stylegan_runner.py
class StyleGAN_Movie(Module):

    def __init__(self, model, length, interp, duration, movie):
        self.model = model              # path to input model
        self.length = length            # (int) number of way points interpolated between (default = 10)
        self.interp = interp            # (int) number of interpolations between waypoints (default = 30)
        self.duration = duration        # (int) duration of animation (default=1)
        self.movie = movie
        self.ready = checkFiles(model)
        self.output_files = [movie]

    def run(self):
        prep_output_dir(self.movie)
        # Run the stylegan program in a separate sub-process
        cwd = os.getcwd()
        params = {'model': os.path.join(cwd, self.model),
                  'movie_out' : os.path.join(cwd, self.movie, 'movie.gif'),
                  'num' : self.length,
                  'interp' : self.interp,
                  'duration' : self.duration
                 }
        command = 'python stylegan/stylegan_runner.py --movie'
        for key in params.keys():
          val = params[key]
          command = command + ' --' + str(key) + ' ' + str(val)
        print("launching", command)
        try:
          process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True )
          for line in iter(process.stdout.readline, b''):  # replace '' with b'' for Python 3
            sys.stdout.write(line)
        except Exception as e:
          print("something broke")
          print(e)




######################

class MakeMovie(Module):

    def __init__(self, images, duration, movie):
        self.images = images            # path to directory with images
        self.duration = duration        # (int) duration of animation (default=10)
        self.movie = movie              # path to output movie
        self.ready = checkFiles(images)
        self.output_files = [movie]

    def run(self):
        images = []
        files = []
        # Get list of filenames
        for file in os.listdir(self.images):
            files.append(file)
        # Sort the file names by creation date
        sorted_files = sorted(files)
        # Load each image in sorted order
        for file in sorted_files:
            try:
                img = Image.open(os.path.join(self.images, file))
                images.append(img)
            except:
                print(file, "did not load.")
        # Prep the destination, a directory to save a single file
        prep_output_dir(self.movie)
        # Make and save the animation
        images[0].save(os.path.join(self.movie, 'movie.gif'), "GIF",
                       save_all=True,
                       append_images=images[1:],
                       duration=self.duration,
                       loop=0) 

##########################

class Degridify(Module):

    def __init__(self, input, columns, rows, output):
        self.input = input              # path to directory containing images
        self.output = output            # path to output directory
        self.columns = columns          # (int) number of columns
        self.rows = rows                # (int) number of rows
        self.ready = checkFiles(input)
        self.output_files = [output]

    def run(self):
        columns = self.columns
        # iterate through images in the directory
        for file in os.listdir(self.input):
            # Load image
            grid = Image.open(os.path.join(self.input, file))
            # compute the size of each grid cell
            grid_width, grid_height = grid.size
            image_width = grid_width // columns
            image_height = grid_height // rows
            # Start cropping
            for i in range(columns):
                for j in range(rows):
                    img = grid.crop((i*image_width, j*image_height, (i+1)*image_width, (j+1)*image_height))
                    img.save(os.path.join(self.output, str(i)+'-'+str(j)+'.jpg'), "JPEG")

class Gridify(Module):

    def __init__(self, input, size, columns, output):
        self.input = input              # path to directory containing input files
        self.output = ouput             # path containing a grid file
        self.size = size                # (int) size of a cell (default=256)
        self.columns = columns          # (int) number of columns required (rows computed automatically) (default=4)
        self.ready = checkFiles(input)
        self.output_files = [output]

    def run(self):
        columns = self.columns # Just making it easier to use this variable
        # load and resize images
        images = [] # all the images
        files = [] # all the filenames
        p = transforms.Compose([transforms.Resize((self.size,self.size))]) # the transform
        # Sort the files by creation date
        for file in os.listdir(self.input):
            files.append(file)
        sorted_files = sorted(files)
        # Iterate through all images in the directory
        for file in sorted_files:
            # Load the image
            img = Image.open(os.path.join(self.input, file))
            # Resize it
            img2 = p(img)
            # Save it in order
            images.append(img2)
        # compute number of rows
        rows = len(images) // columns
        leftovers = len(images) % columns
        # round up
        if leftovers > 0:
            rows = rows + 1
            for _ in range(leftovers):
                empty = Image.new('RGB', (self.size, self.size))
                images.append(empty)
        # make new image
        grid = Image.new('RGB', (columns*self.size, rows*self.size))
        # paste images into new image
        counter = 0
        for i in range(columns):
            for j in range(rows):
                cur_img = images[counter]
                grid.paste(cur_img, (i*self.size, j*self.size))
                counter = counter + 1
        # save new image
        prep_output_dir(self.output)
        grid.save(os.path.join(self.output, 'grid.jpg'), "JPEG")


###########################


