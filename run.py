import json
import os
import copy
from modules import *

def runModule(module, rest):
	params = rest.replace('{', '').replace('}', '')
	#params = re.sub(r'u\'', '', params)
	#params = re.sub(r'\'', '', params)
	#params = re.sub(r': ', '=', params)
	p1 = re.compile('u\'([a-zA-Z\_]+)\'[\s]*:[\s]*u(\'[0-9a-zA-Z\_\.\/:\*\-]*\')')
	params = p1.sub(r'\1=\2', params)
	params = re.sub('\'True\'', 'True', params)
	params = re.sub('\'False\'', 'False', params)
	p2 = re.compile('\'([0-9]*[\.]*[0-9]+)\'')
	params = p2.sub(r'\1', params)
	evalString = module + '(' + params + ')'
	print evalString
	module = eval(evalString)
	module.run()

data = []
for line in open('run', "r"):
	data.append(json.loads(line))

for d in data:
	module = d['module']
	d2 = copy.deepcopy(d)
	del d2['module']
	runModule(module, str(d2))

