
import os
import re
import shutil

########################
### GLOBALS

NEWLINE = '\n'
INFINITY = float("INFINITY")


########################


def checkFiles(*filenames):
    for filename in filenames:
        if not os.path.exists(filename):
            return False
    return True


#########################

def convertHexToASCII(str):
    return re.sub(r'\%([A-Z0-9][A-Z0-9])', lambda match: "{0}".format(bytes.fromhex(match.group(1)).decode("utf-8")), str) 

#########################

def prep_output_dir(path, makedir=True):
    # Does the directory already exist?
    if os.path.exists(path):
        # it does exist... delete        
        if os.path.isdir(path):
            shutil.rmtree(path)
        else: 
            os.remove(path)
    # Make the directory
    if makedir:
        os.mkdir(path)

#######################
### MODULE BASE CLASS

class Module():
	
	ready = True
	output_files = []

	def run():
		pass
