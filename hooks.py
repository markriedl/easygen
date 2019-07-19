import os
import shutil
from pathlib import Path
from PIL import Image
from IPython.display import display


def python_cwd_hook_aux(dir):
  result = {}
  for file in os.listdir(dir):
    path = os.path.join(dir, file)
    if os.path.isdir(path):
      file = file + '/'
    result[file] = path
  result['./'] = os.getcwd()
  if dir != '/':
    parent_dir = str(Path(dir).parent)
    result['../'] = parent_dir
  return result

def python_move_hook_aux(path1, path2):
  status = False
  if os.path.exists(path1):
    shutil.move(path1, path2)
    status = True
  return status
  
def python_copy_hook_aux(path1, path2):
  status = False
  if os.path.exists(path1):
    # path 1 exists
    if os.path.isdir(path1):
      # path1 is a directory
      if os.path.exists(path2):
        # path2 exists
        if os.path.isdir(path2):
          # copying directory to selected directory
          # make new directory inside with same name as path1
          basename1 = os.path.basename(os.path.normpath(path1))
          path2 = os.path.join(path2, basename1)
          shutil.copytree(path1, path2)
          status = True
        else:
          # copy directory to a file
          # can't do that
          print("cannot copy directory inside a file")
    else:
      # path1 is a file
      # doesn't matter if path2 is a file or directory
      shutil.copy(path1, path2)
      status = True
  return status

def python_open_text_hook_aux(path):
  status = False
  if os.path.exists(path) and not os.path.isdir(path):
    with open(path, 'r') as file:
      try:
        text = file.read()
        print(text)
        status = True
      except:
        print("Cannot read text file", path)
  return status
  
def python_open_image_hook_aux(path):
  status = False
  if os.path.exists(path) and not os.path.isdir(path):
    try:
      pil_im = Image.open(path, 'r')
      display(pil_im)
      status = True
    except:
      print("Cannot open image file", path)
  return status

def python_save_hook_aux(file_text, filename):
  status = False
  with open(filename, 'w') as f:
    try:
      f.write(file_text)
      status = True
    except:
      print("Could not write to", filename)
  return status

def python_load_hook_aux(filename):
  file_text = ''
  with open(filename, 'r') as f:
    try:
      file_text = f.read()
      status = True
    except:
      print("could not write to", filename)
  return file_text

def python_mkdir_hook_aux(path, dir_name):
  status = False
  try:
    os.mkdir(os.path.join(path, dir_name))
    status = True
  except:
    print("Could not create directory " + dir_name + " in " + path)
  return status
