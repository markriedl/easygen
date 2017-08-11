import re
import os
import random
from glob import glob
from module import *
#import readWikipedia
#import lstm
#import seq2seq_translate
#import DCGAN.main as gan
from shutil import copyfile
from shutil import move
import urllib2
import time
import imageio # This needs to be installed first 'pip install imageio'

########################

CHECKPOINT_DIR = 'checkpoints'

########################

def checkFiles(*filenames):
	for filename in filenames:
		if not os.path.exists(filename):
			return False
	return True


#########################

def convertHexToACII(str):
	return re.sub(r'\%([A-Z0-9][A-Z0-9])', lambda match: "{0}".format(match.group(1).decode("hex")), str)


#######################

class ReadTextFile(Module):

	def __init__(self, file, output):
		self.file = file
		self.output = output
		if checkFiles(file):
			self.ready = True

	def run(self):
		copyfile(self.file, self.output)


#################

class WriteTextFile(Module):

	def __init__(self, file, input):
		self.file = file
		self.input = input
		self.ready = checkFiles(input)

	def run(self):
		copyfile(self.input, self.file)


###################

class ConcatenateTextFiles(Module):

	def __init__(self, input_1, input_2, output):
		self.input_1 = input_1
		self.input_2 = input_2
		self.output = output
		if checkFiles(input_1, input_2):
			self.ready = True


	def run(self):
		with open(self.output, 'w') as outfile:
			for line in open(self.input_1, 'rU'):
				print >> outfile, line.strip()
			for line in open(self.input_2, 'rU'):
				print >> outfile, line.strip()


########################

class SplitSentences(Module):

	def __init__(self, input, output):
		self.input = input
		self.output = output
		if checkFiles(input):
			self.ready = True

	def run(self):
		parser = re.compile(r'([\?\.\!:])')
		with open(self.output, 'w') as outfile:
			for line in open(self.input, 'rU'):
				lines = [s.strip() for s in parser.sub(r'\1\n', line).splitlines()]
				for sent in lines:
					if len(sent) > 0:
						print >> outfile, sent

#######

class RemoveEmptyLines(Module):

	def __init__(self, input, output):
		self.input = input
		self.output = output
		if checkFiles(input):
			self.ready = True

	def run(self):
		with open(self.output, 'w') as outfile:
			for line in open(self.input, 'rU'):
				if len(line.strip()) > 0:
					print >> outfile, line.strip()


############

class StripLines(Module):

	def __init__(self, input, output):
		self.input = input
		self.output = output
		if checkFiles(input):
			self.ready = True

	def run(self):
		with open(self.output, 'w') as outfile:
			for line in open(self.input, 'rU'):
				if len(line.strip()) > 0:
					print >> outfile, line.strip()


##############

class ReplaceCharacters(Module):

	def __init__(self, input, output, find, replace):
		self.input = input
		self.output = output
		self.find = find
		self.replace = replace
		if checkFiles(input):
			self.ready = True

	def run(self):
		with open(self.output, 'w') as outfile:
			for line in open(self.input, 'rU'):
				print >> outfile, line.replace(self.find, self.replace)


##################

class MakeTrainTestData(Module):

	def __init__(self, data, training_percent, training_data, testing_data):
		self.data = data
		self.training_percent = training_percent
		self.training_data = training_data
		self.testing_data = testing_data
		if checkFiles(data):
			self.ready = True

	def run(self):
		lines = []
		for line in open(self.data, 'rU'):
			lines.append(line)
		cut = int(len(lines)*self.training_percent)
		with open(self.training_data, 'w') as outfile:
			for i in range(cut):
				print >> outfile, lines[i]
		with open(self.testing_data, 'w') as outfile:
			for i in range(len(lines)-cut):
				print >> outfile, lines[cut+i] 


class MakeTransTrainTestData(Module):

	def __init__(self, data_x, data_y, training_percent, training_x_data, training_y_data, testing_x_data, testing_y_data):
		self.data_x = data_x
		self.data_y = data_y
		self.training_percent = training_percent
		self.training_x_data = training_x_data
		self.training_y_data = training_y_data
		self.testing_x_data = testing_x_data
		self.testing_y_data = testing_y_data
		if checkFiles(data_x, data_y):
			self.ready = True

	def run(self):
		lines1 = []
		lines2 = []
		for line in open(self.data_x, 'rU'):
			lines1.append(line)
		for line in open(self.data_y, 'rU'):
			lines2.append(line)
		cut = int(len(lines1)*self.training_percent)
		with open(self.training_x_data, 'w') as outfile:
			for i in range(cut):
				print >> outfile, lines1[i]
		with open(self.training_y_data, 'w') as outfile:
			for i in range(cut):
				print >> outfile, lines2[i]
		with open(self.testing_x_data, 'w') as outfile:
			for i in range(len(lines1)-cut):
				print >> outfile, lines1[cut+i]
		with open(self.testing_y_data, 'w') as outfile:
			for i in range(len(lines2)-cut):
				print >> outfile, lines2[cut+i] 

########################

class ReadWikipedia(Module):

	def __init__(self, wiki_directory, pattern, categories, out_file, titles_file, break_sentences = False):
		self.wiki_directory = wiki_directory
		self.pattern = pattern
		self.out_file = out_file
		self.titles_file = titles_file
		self.break_sentences = break_sentences
		self.categories = categories
		if checkFiles(wiki_directory):
			self.ready = True

	def run(self):
		import readWikipedia
		readWikipedia.ReadWikipedia(self.wiki_directory, self.pattern, self.categories, self.out_file, self.titles_file, self.break_sentences)


#########################

class SplitLines(Module):

	def __init__(self, input, output1, output2, character):
		self.input = input
		self.output1 = output1
		self.output2 = output2
		self.character = character
		if checkFiles(input):
			self.ready = True

	def run(self):
		data1 = []
		data2 = []
		for line in open(self.input, 'rU'):
			splitLine = line.split(self.character, 1)
			data1.append(splitLine[0])
			if len(splitLine) > 1:
				data2.append(splitLine[1])
			else:
				data2.append('')
		with open(self.output1, 'w') as outfile1:
			for line in data1:
				print >> outfile1, line.strip()
		with open(self.output2, 'w') as outfile2:
			for line in data2:
				print >> outfile2, line.strip()

#########################

class MakePredictionData(Module):

	def __init__(self, data, x, y):
		self.data = data
		self.x = x
		self.y = y
		if checkFiles(data):
			self.ready = True

	def run(self):
		last = None
		x_data = []
		y_data = []
		for line in open(self.data, 'rU'):
			if last is not None:
				x_data.append(last.strip())
				y_data.append(line.strip())
			last = line
		with open(self.x, 'w') as xoutfile:
			for line in x_data:
				print >> xoutfile, line
		with open(self.y, 'w') as youtfile:
			for line in y_data:
				print >> youtfile, line

#########################

'''
class MakeLSTMDictionary(Module):

	def __init__(self, data, dictionary, history = 25):
		self.data = data
		self.dictionary = dictionary
		self.history = history

	def run(self):
		lstm.MakeLSTMDictionary(self.data, self.dictionary, self.history)
'''

##########################

class CharRNN_Train(Module):

	def __init__(self, data, model, dictionary, history = 25, layers = 3, epochs = 10, hidden_nodes = 512):
		self.data = data
		self.model = model
		self.dictionary = dictionary
		self.history = history
		self.layers = layers
		self.epochs = epochs
		self.hidden_nodes = hidden_nodes
		#self.dropout = dropout
		if checkFiles(data):
			self.ready = True

	def run(self):
		import lstm
		lstm.CharacterLSTM_Train(self.data, self.model, self.dictionary, self.history, self.layers, self.epochs, self.hidden_nodes)
		copyfile('temp/checkpoint', self.model)

##########################

class CharRNN_Train_More(Module):

	def __init__(self, data, in_model, out_model, dictionary, history = 25, layers = 3, epochs = 10, hidden_nodes = 512):
		self.data = data
		self.in_model = in_model
		self.out_model = out_model
		self.dictionary = dictionary
		self.history = history
		self.layers = layers
		self.epochs = epochs
		self.hidden_nodes = hidden_nodes
		#self.dropout = dropout
		if checkFiles(data, in_model, directory):
			self.ready = True

	def run(self):
		import lstm
		lstm.CharacterLSTM_Train_More(self.data, self.in_model, self.dictionary, self.out_model, self.history, self.layers, self.epochs, self.hidden_nodes)
		copyfile('temp/checkpoint', self.out_model)

#########################

class CharRNN_Run(Module):

	def __init__(self, seed, dictionary, model, output, steps = 600, layers = 3, hidden_nodes = 512, history = 25, temperature = 0.5):
		self.seed = seed
		self.dictionary = dictionary
		self.model = model
		self.steps = steps
		self.layers = layers
		self.hidden_nodes = hidden_nodes
		self.history = history
		self.temperature = temperature
		self.output = output
		if checkFiles(seed, dictionary, model):
			self.ready = True

	def run(self):
		import lstm		
		file = open(self.seed, 'r') 
		seed = file.read() 
		file.close()

		copyfile(self.model, 'temp/checkpoint')

		result = lstm.CharacterLSTM_Run(seed, self.dictionary, self.model, self.output, self.steps, self.layers, self.hidden_nodes, self.history, self.temperature)

		with open(self.output, 'w') as outfile:
			print >> outfile, result

############################


class Seq2Seq_Train(Module):

	def __init__(self, all_data, x, y, model, dictionary, layers, hidden_nodes, epochs):
		self.all_data = all_data
		self.model = model
		self.layers = layers
		self.hidden_nodes = hidden_nodes
		self.epochs = epochs
		self.x = x
		self.y = y
		self.dictionary = dictionary
		if checkFiles(all_data, x, y):
			self.ready = True

	def run(self):
		import seq2seq_translate
		# read data
		data = []
		for line in open(self.all_data, 'rU'):
			data.append(line.strip())
		x_data = []
		for line in open(self.x, 'rU'):
			x_data.append(line.strip())
		y_data = []
		for line in open(self.y, 'rU'):
			y_data.append(line.strip())

		# split into training and validation 
		data_split = int(len(data) * 0.9)
		train = data[0:data_split]
		validation = data[data_split:]

		x_split = int(len(x_data) * 0.9)
		train_x = x_data[0:x_split]
		validation_x = x_data[x_split:]
		
		y_split = int(len(y_data) * 0.9)
		train_y = y_data[0:y_split]
		validation_y = y_data[y_split:]

		name = self.all_data.split('/')[1]
		out_name = self.model.split('/')[1]

		print('writing ' + name + '.train')
		f = open('temp/' + name + '.train', 'w')
		for line in train:
			print >> f, line
		f.close()

		print('writing ' + name + '.validation')
		f = open('temp/' + name + '.validation', 'w')
		for line in validation:
			print >> f, line
		f.close()

		print('writing ' + name + '.train.input')
		f = open('temp/' + name + '.train.input', 'w')
		for line in train_x:
			print >> f, line
		f.close()

		print('writing ' + name + '.train.output')
		f = open('temp/' + name + '.train.output', 'w')
		for line in train_y:
			print >> f, line        
		f.close()

		print('writing ' + name + '.validation.input')
		f = open('temp/' + name + '.validation.input', 'w')
		for line in validation_x:
			print >> f, line        
		f.close()

		print('writing ' + name + '.validation.output')
		f = open('temp/' + name + '.validation.output', 'w')
		for line in validation_y:
			print >> f, line        
		f.close()

		seq2seq_translate.train(input_name = name, output_name = out_name, data_dir = 'temp', num_layers = self.layers, size = self.hidden_nodes, max_epochs = self.epochs)
		copyfile('temp/checkpoint', self.model)
		copyfile(self.all_data+'.vocab', self.dictionary)

#####################################

class Seq2Seq_Train_More(Module):

	def __init__(self, all_data, x, y, model_in, dictionary, model_out, layers, hidden_nodes, epochs):
		self.all_data = all_data
		self.model_in = model_in
		self.model_out = model_out
		self.layers = layers
		self.hidden_nodes = hidden_nodes
		self.epochs = epochs
		self.x = x
		self.y = y
		self.dictionary = dictionary
		if checkFiles(all_data, x, y, model_in, dictionary):
			self.ready = True

	def run(self):
		import seq2seq_translate
		# read data
		data = []
		for line in open(self.all_data, 'rU'):
			data.append(line.strip())
		x_data = []
		for line in open(self.x, 'rU'):
			x_data.append(line.strip())
		y_data = []
		for line in open(self.y, 'rU'):
			y_data.append(line.strip())

		# split into training and validation 
		data_split = int(len(data) * 0.9)
		train = data[0:data_split]
		validation = data[data_split:]

		x_split = int(len(x_data) * 0.9)
		train_x = x_data[0:x_split]
		validation_x = x_data[x_split:]
		
		y_split = int(len(y_data) * 0.9)
		train_y = y_data[0:y_split]
		validation_y = y_data[y_split:]

		name = self.all_data.split('/')[1]
		out_name = self.model_out.split('/')[1]

		print('writing ' + name + '.train')
		f = open('temp/' + name + '.train', 'w')
		for line in train:
			print >> f, line
		f.close()

		print('writing ' + name + '.validation')
		f = open('temp/' + name + '.validation', 'w')
		for line in validation:
			print >> f, line
		f.close()

		print('writing ' + name + '.train.input')
		f = open('temp/' + name + '.train.input', 'w')
		for line in train_x:
			print >> f, line
		f.close()

		print('writing ' + name + '.train.output')
		f = open('temp/' + name + '.train.output', 'w')
		for line in train_y:
			print >> f, line        
		f.close()

		print('writing ' + name + '.validation.input')
		f = open('temp/' + name + '.validation.input', 'w')
		for line in validation_x:
			print >> f, line        
		f.close()

		print('writing ' + name + '.validation.output')
		f = open('temp/' + name + '.validation.output', 'w')
		for line in validation_y:
			print >> f, line        
		f.close()

		split_model_path = os.path.split(self.model_in)
		model_name = split_model_path[1]
		model_directory = split_model_path[0]
		if len(model_directory) == 0:
			model_directory = '.'

		#split_model_path = os.path.split(self.model)
		#model_file = split_model_path[1]
		#model_directory = split_model_path[0]

		for f in os.listdir(model_directory):
			match = re.match(model_name, f)
			if match is not None:
				f_rest = f[len(model_name):]
				copyfile(os.path.join(model_directory, f), os.path.join(CHECKPOINT_DIR, f))

		with open(os.path.join(CHECKPOINT_DIR, 'checkpoint'), 'w') as f:
			print >> f, 'model_checkpoint_path: "'+ model_name + '"'
			print >> f, 'all_model_checkpoint_paths: "' + model_name + '"'

		copyfile(self.dictionary, self.all_data+'.vocab')

		seq2seq_translate.train(input_name = name, output_name = out_name, data_dir = 'temp', num_layers = self.layers, size = self.hidden_nodes, max_epochs = self.epochs)
		copyfile('temp/checkpoint', self.model_out)
		#copyfile(self.all_data+'.vocab', self.dictionary)

#######################################

class Seq2Seq_Run(Module):

	def __init__(self, model, data, dictionary, layers, hidden_nodes, stop, output):
		self.model = model
		self.data = data
		self.dictionary = dictionary
		self.layers = layers
		self.hidden_nodes = hidden_nodes
		self.stop = stop
		self.output = output
		if checkFiles(model, data, dictionary):
			self.ready = True


	def run(self):
		import seq2seq_translate
		the_data = []
		for line in open(self.data, 'rU'):
			the_data.append(line.strip())

		name = self.data.split('/')[1]

		print('writing ' + name + '.test')
		f = open('temp/' + name + '.test', 'w')
		for line in the_data:
			print >> f, line
		f.close()

		copyfile(self.model, 'temp/checkpoint')
		copyfile(self.dictionary, self.data+'.vocab')
		results = seq2seq_translate.decode(name = name, data_dir = 'temp', stop_symbol = self.stop, num_layers = self.layers, size = self.hidden_nodes)

		with open(self.output, 'w') as outfile:
			for line in results:
				print >> outfile, line


##########################################

class RandomSequence(Module):

	def __init__(self, input, output, length):
		self.input = input
		self.output = output
		self.length = length
		if checkFiles(input):
			self.ready = True

	def run(self):
		sequence = ''
		import lstm
		sequence = lstm.random_sequence_from_textfile(self.input, self.length)
		with open(self.output, 'w') as f:
			print >> f, sequence.strip()

############################################

class MakeString(Module):

	def __init__(self, string, output):
		self.string = string
		self.output = output

	def run(self):
		with open(self.output, 'w') as f:
			print >> f, self.string.strip()



###############################################

class UserInput(Module):

	def __init__(self, prompt, output):
		self.prompt = prompt
		self.output = output

	def run(self):
		prompt = self.prompt
		if len(self.prompt) == 0:
			prompt = 'prompt: '
		s = raw_input(prompt)

		with open(self.output, 'w') as f:
			print >> f, s

###################################################

class ReadImages(Module):

	def __init__(self, data_directory, output_images):
		self.data_directory = data_directory
		self.output_images = output_images
		if checkFiles(data_directory):
			self.ready = True

	def run(self):
		with open(self.output_images, 'w') as f:
			print >> f, self.data_directory

###################################################

class WriteImages(Module):

	def __init__(self, input_images, output_directory):
		self.input_images = input_images
		self.output_directory = output_directory

	def run(self):
		f = open(self.input_images, 'r')
		data_dir = f.readline().strip()
		f.close()

		if not os.path.exists(self.output_directory):
			os.makedirs(self.output_directory)
		
		for item in os.listdir(data_dir):
			s = os.path.join(data_dir, item)
			d = os.path.join(self.output_directory, item)
			copyfile(s, d)

###################################################



class DCGAN_Train(Module):

	def __init__(self, input_images, animation, epochs, input_height, output_height, filetype, model):
		self.input_images = input_images
		self.epochs = epochs
		self.input_height = input_height
		self.output_height = output_height
		self.filetype = filetype
		#self.crop = crop
		#self.output_images = output_images
		#self.num_images = num_images
		self.animation = animation
		self.model = model
		if checkFiles(input_images):
			self.ready = True

	def run(self):
		import DCGAN.main as gan
		filetype = ''
		if len(self.filetype) == 0:
			filetype = '*.jpg'
		else:
			filetype = '*.' + self.filetype
		#crop = self.crop
		#if isinstance(self.crop, basestring):
		#	crop = True
		# Get data directory
		f = open(self.input_images, 'r')
		data_dir = f.readline().strip()
		f.close()
		# Create directory for samples and output and checkpoints
		#output_dir = self.output_images + '_output'
		sample_dir = self.model + '_samples'
		checkpoint_dir = CHECKPOINT_DIR
		#if not os.path.exists(output_dir):
		#	os.makedirs(output_dir)
		if not os.path.exists(sample_dir):
			os.makedirs(sample_dir)
		# Figure out where to save the model
		model_path_split = os.path.split(self.model)
		model_dir = model_path_split[0]
		model_filename = model_path_split[1]
		if len(model_dir) == 0:
			model_dir = '.'
		gan.train(epoch = self.epochs, input_height = self.input_height, input_width = self.input_height, output_height = self.output_height, output_width = self.output_height, input_fname_pattern = filetype, dataset = data_dir, sample_dir = sample_dir, checkpoint_dir = checkpoint_dir, model_dir = model_dir, model_filename = model_filename)

		#with open(self.output_images, 'w') as f:
		#	print >> f, output_dir

		# Make the training animation
		images = []
		samples = sorted(os.listdir(sample_dir), key=lambda f: os.stat(os.path.join(sample_dir, f)).st_mtime)
		for sample in samples:
		    images.append(imageio.imread(os.path.join(sample_dir, sample), 'png'))
		if  len(images) > 0:
			imageio.mimsave(self.animation, images, 'gif')

#####################################################

class DCGAN_Run(Module):

	def __init__ (self, input_images, input_height, output_height, filetype, output_image, model):
		self.input_images = input_images
		self.output_image = output_image
		self.input_height = input_height
		self.output_height = output_height
		self.filetype = filetype
		#self.num_images = num_images
		self.model = model
		self.ready = checkFiles(model, input_images)

	def run(self):
		import DCGAN.main as gan
		filetype = ''
		if len(self.filetype) == 0:
			filetype = '*.jpg'
		else:
			filetype = '*.' + self.filetype
		# Get data directory
		f = open(self.input_images, 'r')
		data_dir = f.readline().strip()
		f.close()
		# Run the model
		copyfile(self.model, 'temp/checkpoint')
		gan.run(output_dir = self.output_image, input_height = self.input_height, input_width = self.input_height, output_height = self.output_height, output_width = self.output_height, input_fname_pattern = filetype, dataset = data_dir, checkpoint_dir = 'temp')


####################################################

'''
class ParseWords(Module):

	def __init__(self, input, output):
		from stanford_corenlp_python import StanfordNLP # https://github.com/dasmith/stanford-corenlp-python
		self.input = input
		self.output = output

	def run(self):
		stanfordParser = StanfordNLP()
		print stanfordParser

		lines = []
		new_lines = []
		final_lines = []
		for line in open(self.input, 'rU'):
			lines.append(line)

		parser = re.compile(r'([\?\.\!:])')
		for line in lines:
			split_lines = [s.strip() for s in parser.sub(r'\1\n', line).splitlines()]
			for sent in split_lines:
				if len(sent) > 0:
					new_lines.append(sent)

		for line in new_lines:
			line = re.sub(r'\xe2\x80\x94', '-', line)
			print line
			try:
				result = stanfordParser.parse(line)
				sentences = result['sentences']
				for sent_struct in sentences:
					words = sent_struct['words']
					parsed = map(lambda x: x[0], words)
					text = ''
					for word in parsed:
						text = text + word + ' '
					print text
					final_lines.append(text.strip())
			except:
				print "error"
				
		with open(self.output, 'w') as outfile:
			for line in lines:
				print >> outfile, line
'''

####################################

class PickFromWikipedia(Module):

	def __init__(self, wiki_directory, input, categories, section_name, output, break_sentences):
		self.wiki_directory = wiki_directory # path to wikipedia dump files
		self.input = input # name of file. Each line should be a title of a wikipedia article
		self.output = output # name of file. Paragraphs pulled from wikipedia. <EOS> after each article.
		self.section_name = section_name # Do you just want a sub-section? Or '' for all.
		self.break_sentences = break_sentences # bool - one sentence per line?
		self.categories = categories # Restrict to certain categories
		if checkFiles(wiki_directory, input):
			self.ready = True

	def run(self):
		import readWikipedia
		pattern = ''
		first = True
		for line in open(self.input, 'rU'):
			line = line.strip()
			if first:
				if len(line) > 0:
					pattern = line
				first = False
			else:
				if len(line) > 0:
					pattern = pattern + '|' + line
		if len(self.section_name.strip()) > 0:
			pattern = pattern + ':' + self.section_name.strip()
		print pattern
		readWikipedia.ReadWikipedia(self.wiki_directory, pattern, self.categories, self.output, 'temp/pickfromwikipediatitles', self.break_sentences)

#######################################

class RandomizeLines(Module):

	def __init__(self, input, output):
		self.input = input
		self.output = output
		if checkFiles(input):
			self.ready = True

	def run(self):
		lines = []
		for line in open(self.input, 'rU'):
			lines.append(line)
		random.shuffle(lines)
		with open(self.output, 'w') as outfile:
			for line in lines:
				print >> outfile, line.strip()


#########################################

class RemoveTags(Module):

	def __init__(self, input, output):
		self.input = input
		self.output = output
		if checkFiles(input):
			self.ready = True

	def run(self):
		lines = []
		for line in open(self.input, 'rU'):
			line = re.sub('<[^<]+?>', '', line)
			if len(line) > 0:
				lines.append(line.strip())
		with open(self.output, 'w') as outfile:
			for line in lines:
				print >> outfile, line					


##############################

class MakeLowercase(Module):

	def __init__(self, input, output):
		self.input = input
		self.output = output
		if checkFiles(input):
			self.ready = True

	def run(self):
		with open(self.output, 'w') as outfile:
			for line in open(self.input, 'rU'):
				print >> outfile, line.lower().strip()

################################

class Wordify(Module):

	def __init__(self, input, output):
		self.input = input
		self.output = output
		if checkFiles(input):
			self.ready = True

	def run(self):
		parser = re.compile(r'([\,\.\?\!\'\"\(\)\[\]\{\}])')
		with open(self.output, 'w') as outfile:
			for line in open(self.input, 'rU'):
				line = parser.sub(r' \1 ', line)
				line = re.sub('[  ][ ]*', ' ', line) # might put too many spaces in
				print >> outfile, line.strip()


#####################################

class SaveModel(Module):

	def __init__(self, model, file):
		self.model = model
		self.file = file

	def run(self):
		split_model_path = os.path.split(self.model)
		model_directory = split_model_path[0]
		model_file = split_model_path[1]

		split_file_path = os.path.split(self.file)
		target_file = split_file_path[1]
		target_directory = split_file_path[0]
		if len(target_directory) == 0:
			target_directory = '.'

		for f in os.listdir(model_directory):
			match = re.match(model_file, f)
			if match is not None:
				f_rest = f[len(model_file):]
				filename = os.path.join(model_directory, f)
				if not os.path.isdir(filename):
					copyfile(filename, os.path.join(target_directory, target_file) + f_rest)

		with open(self.file, 'w') as f:
			print >> f, 'model_checkpoint_path: "'+ target_file + '"'
			print >> f, 'all_model_checkpoint_paths: "' + target_file + '"'


###################################

class SaveDictionary(Module):

	def __init__(self, dictionary, file):
		self.dictionary = dictionary
		self.file = file

	def run(self):
		copyfile(self.dictionary, self.file)

#####################################

class LoadDictionary(Module):

	def __init__(self, file, dictionary):
		self.dictionary = dictionary
		self.file = file
		if checkFiles(dictionary):
			self.ready = True

	def run(self):
		copyfile(self.file, self.dictionary)

######################################

class LoadModel(Module):

	def __init__(self, file, model):
		self.file = file
		self.model = model
		if checkFiles(file):
			self.ready = True

	def run(self):
		split_file_path = os.path.split(self.file)
		file_name = split_file_path[1]
		file_directory = split_file_path[0]
		if len(file_directory) == 0:
			file_directory = '.'

		split_model_path = os.path.split(self.model)
		model_file = split_model_path[1]
		model_directory = split_model_path[0]

		print file_directory
		print file_name
		for f in os.listdir(file_directory):
			match = re.match(file_name, f)
			if match is not None:
				f_rest = f[len(file_name):]
				outname = self.model + f_rest
				copyfile(os.path.join(file_directory, f), outname)

		with open(self.model, 'w') as f:
			print >> f, 'model_checkpoint_path: "'+ model_file + '"'
			print >> f, 'all_model_checkpoint_paths: "' + model_file + '"'


##########################################

class KeepFirstLine(Module):

	def __init__(self, input, output):
		self.input = input
		self.output = output
		if checkFiles(input):
			self.ready = True

	def run(self):
		with open(self.output, 'w') as outfile:
			for line in open(self.input, 'rU'):
				print >> outfile, line.strip()
				break


###########################################

class DeleteFirstLine(Module):

	def __init__(self, input, output):
		self.input = input
		self.output = output
		if checkFiles(input):
			self.ready = True

	def run(self):
		with open(self.output, 'w') as outfile:
			first = True
			for line in open(self.input, 'rU'):
				if first:
					first = False
				else:
					print >> outfile, line.strip()

##########################################

class DeleteLastLine(Module):

	def __init__(self, input, output):
		self.input = input
		self.output = output
		if checkFiles(input):
			self.ready = True

	def run(self):
		lines = []
		for line in open(self.input, 'rU'):
			lines.append(line)
		with open(self.output, 'w') as outfile:
			for line in lines[0:len(lines)-1]:
				print >> outfile, line

###########################################

class ReadFromWeb(Module):

	def __init__(self, url, data):
		self.url = url
		self.data = data

	def run(self):
		import urllib2
		response = urllib2.urlopen(self.url)
		html = response.read()
		html_lines = html.splitlines()
		with open(self.data, 'w') as outfile:
			for line in html_lines:
				print >> outfile, line.strip()

############################################

class ReadAllFromWeb(Module):

	def __init__(self, urls, data):
		self.urls = urls
		self.data = data
		if checkFiles(urls):
			self.ready = True

	def run(self):
		html_lines = []
		try:
			for url in open(self.urls, 'rU'):
				response = urllib2.urlopen(url)
				print "reading", url
				html = response.read()
				html_lines = html_lines + html.splitlines()
				time.sleep(30+random.randint(0,10))
		except urllib2.HTTPError:
			print "Service unavailable"
		with open(self.data, 'w') as outfile:
			for line in html_lines:
				print >> outfile, line.strip()

###########################################

class KeepLineWhen(Module):

	def __init__(self, input, match, output):
		self.input = input
		self.output = output
		self.match = convertHexToACII(match)
		if checkFiles(input):
			self.ready = True

	def run(self):
		lines = []
		with open(self.output, 'w') as outfile:
			for line in open(self.input, 'rU'):
				m = re.search(self.match, line)
				if m is not None:
					print >> outfile, line

##########################################

class KeepLineUnless(Module):

	def __init__(self, input, match, output):
		self.input = input
		self.output = output
		self.match = match
		if checkFiles(input):
			self.ready = True

	def run(self):
		lines = []
		with open(self.output, 'w') as outfile:
			for line in open(self.input, 'rU'):
				m = re.search(self.match, line)
				if m is None:
					print >> outfile, line


#############################################

class MakeCountFile(Module):

	def __init__(self, num, prefix, postfix, output):
		self.num = num
		self.prefix = prefix
		self.postfix = postfix
		self.output = output

	def run(self):
		num = self.num
		if num < 0:
			num = 0

		with open(self.output, 'w') as outfile:
			for n in range(num+1):
				if n > 0:			
					print >> outfile, self.prefix.strip() + str(n) + self.postfix.strip()

############################################
## This probably doesn't work very well

class SplitHTML(Module):

	def __init__(self, input, output):
		self.input = input
		self.output = output
		if checkFiles(input):
			self.ready = True

	def run(self):
		file = open(self.input, 'rU')
		text = file.read()
		file.close()
		lines = []
		can_close = True
		parser = re.compile(r'<[\S\s]+?>')
		while len(text) > 0:
			match = parser.match(text)
			if match is not None:
				tag = match.group(0)
				if tag[1] == '/' and can_close:
					lines[-1] += tag
					can_close = False
				else :
					lines.append(tag)
					can_close = True
				text = text[match.end(0):]
			elif '<' in text:
				i = text.index('<')
				temp = text[0:i]
				if len(lines) > 0:
					lines[-1] += temp
				else:
					lines.append(temp)
				text = text[i:]
			else:
				if len(lines) > 0:
					lines[-1] += text
				else:
					lines.append(text)
				text = ''
		with open(self.output, 'w') as outfile:
			for line in lines:
				print >>outfile, line

###########################################

class Regex_Search(Module):

	def __init__(self, input, expression, output, group_1, group_2):
		self.input = input
		self.expression = expression
		self.output = output
		self.group_1 = group_1
		self.group_2 = group_2
		if checkFiles(input):
			self.ready = True

	def run(self):
		file = open(self.input, 'rU')
		text = file.read()
		file.close()

		with open(self.output, 'w') as outfile:
			with open(self.group_1, 'w') as group1file:
				with open(self.group_2, 'w') as group2file:
					for match in re.finditer(self.expression, text):
						num_groups = len(match.groups())
						print num_groups
						print >> outfile, match.group(0)
						if num_groups > 0:
							print >> group1file, match.group(1)
						else:
							print >> group1file, ''
						if num_groups > 1:
							print >> group2file, match.group(2)
						else:
							print >> group2file, ''


#############################################

class Regex_Sub(Module):

	def __init__(self, input, expression, replace, output):
		self.input = input
		self.expression = expression
		self.replace = replace
		self.output = output
		if checkFiles(input):
			self.ready = True

	def run(self):
		file = open(self.input, 'rU')
		text = file.read()
		file.close()

		with open(self.output, 'w') as outfile:
			new_text = re.subn(self.expression, self.replace, text)[0]
			outfile.write(new_text)


###################################################

class ReadImageFile(Module):

	def __init__(self, file, output):
		self.file = file
		self.output = output
		self.ready = checkFiles(file)

	def run(self):
		copyfile(self.file, self.output)


###################################################

class WriteImageFile(Module):

	def __init__(self, input, file):
		self.input = input
		self.file = file
		self.ready = checkFiles(input)

	def run(self):
		copyfile(self.input, self.file)


##################################################

class StyleNet_Train(Module):

	def __init__(self, style_image, test_image, model, epochs, animation):
		self.style_image = style_image
		self.test_image = test_image
		#self.output_image = output_image
		self.epochs = epochs
		self.animation = animation
		self.model = model
		self.ready = checkFiles(style_image, test_image)

	def run(self):
		import stylenet.style as sstyle 
		#import stylenet.evaluate as sevaluate
		model_path_split = os.path.split(self.model)
		model_path = model_path_split[0]
		model_name = model_path_split[1]
		test_dir = os.path.join(model_path, model_name + "_sample")
		if not os.path.exists(test_dir):
			os.makedirs(test_dir)
		checkpoint_path = os.path.join(CHECKPOINT_DIR, 'style')
		sstyle.main(self.style_image, self.test_image, epochs = self.epochs, test_dir = test_dir, checkpoint_dir = checkpoint_path)
		#sevaluate.main(self.target_image, self.output_image)

		# Save the model
		saver = SaveModel(model=checkpoint_path, file=self.model)
		saver.run()

		# Make the training animation
		images = []
		samples = sorted(os.listdir(test_dir), key=lambda f: os.stat(os.path.join(test_dir, f)).st_mtime)
		for sample in samples:
		    images.append(imageio.imread(os.path.join(test_dir, sample), 'png'))
		if  len(images) > 0:
			imageio.mimsave(self.animation, images, 'gif')

###################################################

class StyleNet_Run(Module):

	def __init__(self, model, target_image, output_image):
		self.model = model
		self.target_image = target_image
		self.output_image = output_image

	def run(self):
		import stylenet.evaluate as sevaluate
		checkpoint_path = os.path.join(CHECKPOINT_DIR, 'style')
		loader = LoadModel(file=self.model, model=checkpoint_path)
		loader.run()
		model_path_split = os.path.split(self.model)
		model_path = model_path_split[0]
		model_name = model_path_split[1]
		with open('temp/checkpoint', 'w') as f:
			print >> f, 'model_checkpoint_path: "style"'
			print >> f, 'all_model_checkpoint_paths: "style"'

		sevaluate.main(self.target_image, self.output_image, checkpoint_dir = checkpoint_path)

##################################			

class Sort(Module):

	def __init__(self, input, output):
		self.input = input
		self.output = output
		self.ready = checkFiles(input);

	def run(self):
		lines = []
		for line in open(self.input, 'rU'):
			lines.append(line)

		sorted_lines = sorted(lines)

		with open(self.output, 'w') as outfile:
			for line in sorted_lines:
				print >> outfile, line.strip()

#####################################

class Reverse(Module):

	def __init__(self, input, output):
		self.input = input
		self.output = output
		self.ready = checkFiles(input);

	def run(self):
		lines = []
		for line in open(self.input, 'rU'):
			lines.append(line)

		lines.reverse()

		with open(self.output, 'w') as outfile:
			for line in lines:
				print >> outfile, line.strip()


#####################################

class MakeEmptyText(Module):

	def __init__(self, output):
		self.output = output
		self.ready = True

	def run(self):
		with open(self.output, 'w') as outfile:
			print >>outfile, ''


########################################

class PrintText(Module):

	def __init__(self, input):
		self.input = input
		self.ready = checkFiles(input)

	def run(self):
		for line in open(self.input, 'rU'):
			print line.strip()




