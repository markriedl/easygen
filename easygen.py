import json
import os
import copy
from modules import *
import argparse

parser = argparse.ArgumentParser(description='Run an easygen program.')
parser.add_argument('program', help="the file containing the easgen program.")
args = parser.parse_args()

def runModule(module_json):
	module_json_copy = copy.deepcopy(module_json)
	if 'module' in module_json_copy:
		module = module_json_copy['module']
		## Convert the json to a set of parameters to pass into a class of the same name as the module name
		## Take the module name out
		del module_json_copy['module']

		rest = str(module_json_copy)

		params = rest.replace('{', '').replace('}', '')
		#params = re.sub(r'u\'', '', params)
		#params = re.sub(r'\'', '', params)
		#params = re.sub(r': ', '=', params)
		p1 = re.compile('u\'([0-9a-zA-Z\_]+)\'[\s]*:[\s]*u(\'[\(\)0-9a-zA-Z\_\.\/:\*\-\?\+\=\&\%\\\[\] ]*\')')
		params = p1.sub(r'\1=\2', params)
		params = re.sub('\'True\'', 'True', params)
		params = re.sub('\'False\'', 'False', params)
		p2 = re.compile('\'([0-9]*[\.]*[0-9]+)\'')
		params = p2.sub(r'\1', params)
		## Put the module name back on as class name
		evalString = module + '(' + params + ')'
		print evalString
		## If everything went well, we can now evaluate the string and create a new class.
		module = eval(evalString)
		## Run the class.
		module.run()

### Make sure required directories exist
temp_directory = './temp'
checkpoints_directory = './checkpoints'
if not os.path.exists(temp_directory):
    os.makedirs(temp_directory)
if not os.path.exists(checkpoints_directory):
    os.makedirs(checkpoints_directory)

### Read in the program file
data_text = ''
data = None
for line in open(args.program, "r"):
	data_text = data_text + line.strip()

### Convert to json dictionary
data = json.loads(data_text)

### Run each module
if data is not None:
	print "Running", args.program
	for d in data:
		runModule(d)
else:
	print "Program is empty"