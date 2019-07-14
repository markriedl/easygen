import re
import os
import pdb
import random
from module import *
import shutil 
import urllib
import time
#import imageio # This needs to be installed first 'pip install imageio'

#######################

class ReadTextFile(Module):

    def __init__(self, file, output):
        self.file = file        # path to file to read
        self.output = output    # path to file to save to
        self.ready = checkFiles(file)
        self.output_files = [output]

    def run(self):
        # copy the file to temp
        if os.path.exists(self.file):
            shutil.copy(self.file, self.output)


#################

class WriteTextFile(Module):

    def __init__(self, input, file):
        self.file = file                  # path of file to write to
        self.input = input                # path of file to write out
        self.ready = checkFiles(input)
        self.output_files = [file]

    def run(self):
        # copy file out of temp to final destination
        if len(self.file) > 0 and os.path.exists(self.input):
            shutil.copy(self.input, self.file)


###################

class ConcatenateTextFiles(Module):

    def __init__(self, input_1, input_2, output):
        self.input_1 = input_1            # path of file 1
        self.input_2 = input_2            # path of file 2
        self.output = output              # path of joined file
        self.ready = checkFiles(input_1, input_2)
        self.output_files = [output]


    def run(self):
        # Open the output file and write the lines from input1 and input2 into it
        with open(self.output, 'w') as outfile:
            for line in open(self.input_1, 'rU'):
                outfile.write(line.strip() + '\n')
            for line in open(self.input_2, 'rU'):
                outfile.write(line.strip() + '\n')


########################

class SplitSentences(Module):

    def __init__(self, input, output):
        self.input = input       # path of file to split
        self.output = output     # path of finalized split
        self.ready = checkFiles(input)
        self.output_files = [output]

    def run(self):
        # Parser will look for relevant punctuation
        parser = re.compile(r'([\?\.\!:])')
        with open(self.output, 'w') as outfile:
            # Read input file
            for line in open(self.input, 'rU'):
                # split up the line.
                lines = [s.strip() for s in parser.sub(r'\1\n', line).splitlines()]
                # write each fragment to file
                for sent in lines:
                    if len(sent) > 0:
                        outfile.write(sent + '\n')

##################################

class RemoveEmptyLines(Module):

    def __init__(self, input, output):
        self.input = input         # path of file to alter
        self.output = output       # path of altered file
        self.ready = checkFiles(input)
        self.output_files = [output]
            
    def run(self):
        with open(self.output, 'w') as outfile:
            for line in open(self.input, 'rU'):
                if len(line.strip()) > 0:
                    outfile.write(line.strip() + '\n')


############

class StripLines(Module):

    def __init__(self, input, output):
        self.input = input       # path of file to strip
        self.output = output     # path of output
        self.ready = checkFiles(input)
        self.output_files = [output]

    def run(self):
        with open(self.output, 'w') as outfile:
            for line in open(self.input, 'rU'):
                if len(line.strip()) > 0:
                    outfile.write(line.strip() + '\n')

############

class TextSubtract(Module):

    def __init__(self, main, subtract, diff):
        self.main = main          # path to text file to subtract lines from
        self.subtract = subtract  # path to text file of lines to subtract from main
        self.diff = diff          # path to result file
        self.ready = checkFiles(main, subtract)
        self.output_files = [diff]

    def run(self):
        lines1 = set()
        lines2 = set()
        for line in open(self.main, 'r'):
            lines1.add(line.strip())
        for line in open(self.subtract, 'r'):
            lines2.add(line.strip())
        lines3 = lines1.difference(lines2)
        with open(self.diff, 'w') as outfile:
            for line in lines3:
                outfile.write(line.strip() + NEWLINE)

###############

class DuplicateText(Module):

    def __init__(self, input, count, output):
        self.input = input    # path of file to modify
        self.output = output  # path of file with duplicates
        self.count = count    # number of times to duplicate the input (int)
        self.ready = checkFiles(input)
        self.output_files = [output]

    def run(self):
        lines = []
        for line in open(self.input, 'r'):
            lines.append(line.strip())
        with open(self.output, 'w') as outfile:
            for _ in range(self.count + 1):
                for line in lines:
                    outfile.write(line.strip() + NEWLINE)

##############

class ReplaceCharacters(Module):

    def __init__(self, input, output, find, replace):
        self.input = input      # path of file to modify
        self.output = output    # path of output
        self.find = find        # string containing substring of characters to replace
        self.replace = replace  # string containing the replacement substring
        self.ready = checkFiles(input)
        self.output_files = [output]

    def run(self):
        with open(self.output, 'w') as outfile:
            for line in open(self.input, 'rU'):
                outfile.write(line.replace(self.find, self.replace) + '\n')


##################

class MakeTrainTestData(Module):

    def __init__(self, data, training_percent, training_data, testing_data):
        self.data = data                           # path to file containing a chunk of data
        self.training_percent = float(training_percent)   # float percentage for training
        self.training_data = training_data         # output path for training data
        self.testing_data = testing_data           # output path of testing data
        self.ready = checkFiles(data)
        self.output_files = [training_data, testing_data]

    def run(self):
        lines = [] # lines of data
        # Read the data
        for line in open(self.data, 'rU'):
            lines.append(line)
        # Find the cut point
        cut = int(len(lines)*(self.training_percent/100.0))
        # Write training portion
        with open(self.training_data, 'w') as outfile:
            for i in range(cut):
                outfile.write(lines[i].strip() + '\n')
        # write testing portion
        with open(self.testing_data, 'w') as outfile:
            for i in range(cut, len(lines)):
                outfile.write(lines[i].strip + '\n')

#################################

class MakeXYTrainTestData(Module):

    def __init__(self, data_x, data_y, training_percent, training_x_data, training_y_data, testing_x_data, testing_y_data):
        self.data_x = data_x                      # path of X data
        self.data_y = data_y                      # path of Y data
        self.training_percent = float(training_percent)  # float portion that is training
        self.training_x_data = training_x_data    # path output for training x
        self.training_y_data = training_y_data    # path output for training y
        self.testing_x_data = testing_x_data      # path output for testing x
        self.testing_y_data = testing_y_data      # path output for testing y
        self.ready = checkFiles(data_x, data_y)
        self.output_files = [training_x_data, training_y_data, testing_x_data, testing_y_data]

    def run(self):
        lines1 = [] # x data
        lines2 = [] # y data
        # Read x data
        for line in open(self.data_x, 'rU'):
            lines1.append(line)
        # Read y data
        for line in open(self.data_y, 'rU'):
            lines2.append(line)
        # find cut point
        cut = int(len(lines1)*self.training_percent)
        # Write training x data
        with open(self.training_x_data, 'w') as outfile:
            for i in range(cut):
                outfile.write(lines1[i].strip() + NEWLINE)
        # Write training y data
        with open(self.training_y_data, 'w') as outfile:
            for i in range(cut):
                outfile.write(lines2[i].strip() + NEWLINE)
        # Write testing x data
        with open(self.testing_x_data, 'w') as outfile:
            for i in range(cut, len(lines1)):
                outfile.write(lines1[i].strip() + NEWLINE)
        # Write testing y data
        with open(self.testing_y_data, 'w') as outfile:
            for i in range(cut, len(lines2)):
                outfile.write(lines2[i].strip() + NEWLINE)

########################

class ReadWikipedia(Module):

    def __init__(self, wiki_directory, pattern, categories, out_file, titles_file):
        self.wiki_directory = wiki_directory # path to find wikipedia files
        self.pattern = pattern               # string of pattern to look for
        self.out_file = out_file             # path of output file
        self.titles_file = titles_file       # path of output titles file
        self.categories = categories         # string of categories
        self.ready = checkFiles(wiki_directory)
        self.output_files = [out_file, titles_file]

    def run(self):
        import readWikipedia
        readWikipedia.ReadWikipedia(self.wiki_directory, self.pattern, self.categories, self.out_file, self.titles_file)


#########################

class SplitLines(Module):

    def __init__(self, input, output1, output2, character):
        self.input = input
        self.output1 = output1
        self.output2 = output2
        self.character = character
        self.ready = checkFiles(input)
        self.output_files = [output1, output2]

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
                outfile1.write(line.strip() + NEWLINE)
        with open(self.output2, 'w') as outfile2:
            for line in data2:
                outfile2.write(line.strip() + NEWLINE)

#########################

'''
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
'''

#########################

class WordRNN_Train(Module):

    def __init__(self, data, dictionary, model, history = 35, layers = 2, epochs = 50, hidden_nodes = 512, learning_rate = 0.0001):
        self.data = data                      # path to training data
        self.model = model                    # path to save model to
        self.dictionary = dictionary          # path to save dictionary to
        self.history = history                # number of words to keep in history
        self.layers = layers                  # number of layers
        self.epochs = epochs                  # number of epochs
        self.learning_rate = learning_rate    # learning rate
        self.hidden_nodes = hidden_nodes      # number of hidden nodes
        self.ready = checkFiles(data)
        self.output_files = [model, dictionary]

    def run(self):
        import lstm
        print("Keyboard interrupt will stop training but program will try to continue.")
        try:
            lstm.wordLSTM_Train(train_data_path=self.data, 
                                dictionary_path=self.dictionary, model_out_path=self.model, 
                                history=self.history, layers=self.layers, epochs=self.epochs, 
                                lr=self.learning_rate)
        except KeyboardInterrupt:
            print("keyboard interrupt")

#########################

class WordRNN_Run(Module):

    def __init__(self, model, dictionary, output, seed, steps = 600, temperature = 0.5, k = 40):
        self.model = model                  # path to model
        self.dictionary = dictionary        # path to dictionary
        self.output = output                # path to write output text to
        self.seed = seed                    # path to a file with a single string
        self.steps = steps                  # number of words to generate
        self.temperature = temperature      # temperature value
        self.k = k                          # top k sampling value
        self.ready = checkFiles(model, dictionary, seed)
        self.output_files = [output]

    def run(self):
        import lstm
        seed = '' # the seed text
        # Read the first line of the file to get the seed
        for line in open(self.seed, 'rU'):
            seed = line.strip()
            break
        lstm.wordLSTM_Run(model_path=self.model, dictionary_path=self.dictionary, output_path=self.output,
                          seed=seed, steps=self.steps, temperature=self.temperature, k = self.k)

############################

class CharRNN_Train(Module):

    def __init__(self, data, dictionary, model, history = 35, layers = 2, epochs = 50, hidden_nodes = 512, learning_rate = 0.0001):
        self.data = data                      # path to training data
        self.model = model                    # path to save model to
        self.dictionary = dictionary          # path to save dictionary to
        self.history = history                # number of characters to keep in history
        self.layers = layers                  # number of layers
        self.epochs = epochs                  # number of epochs
        self.learning_rate = learning_rate    # learning rate
        self.hidden_nodes = hidden_nodes      # number of hidden nodes
        self.ready = checkFiles(data)
        self.output_files = [model, dictionary]

    def run(self):
        import lstm
        print("Keyboard interrupt will stop training but program will try to continue.")
        try:
            lstm.charLSTM_Train(train_data_path=self.data, 
                                dictionary_path=self.dictionary, model_out_path=self.model, 
                                history=self.history, layers=self.layers, epochs=self.epochs, 
                                lr=self.learning_rate)
        except KeyboardInterrupt:
            print("keyboard interrupt")

#########################

class CharRNN_Run(Module):

    def __init__(self, model, dictionary, output, seed, steps = 600, temperature = 0.5):
        self.model = model                  # path to model
        self.dictionary = dictionary        # path to dictionary
        self.output = output                # path to write output text to
        self.seed = seed                    # starting string
        self.steps = steps                  # number of characters to generate
        self.temperature = temperature      # temperature value
        self.ready = checkFiles(model, dictionary, seed)
        self.output_files = [output]

    def run(self):
        import lstm
        seed = '' # the seed text
        # Read the first line of the file to get the seed
        for line in open(self.seed, 'rU'):
            seed = line.strip()
            break
        lstm.charLSTM_Run(model_path=self.model, dictionary_path=self.dictionary, output_path=self.output,
                          seed=seed, steps=self.steps, temperature=self.temperature)

###############################

'''
class GPT2_Load(Module):

    def __init__(self, model_size, model_out):
        self.model_size = model_size # size of GPT-2 model (e.g., "117M")
        self.model_out = model_out # path to model
        self.ready = True
        self.output_files = [model_out]

    def run(self):
        gpt_path = os.path.join(os.getcwd(), "gpt-2", "models", self.model_size)
        if os.path.exists(self.model_out):
            shutil.rmtree(self.model_out)
        shutil.copytree(gpt_path, self.model_out)
'''

class GPT2_FineTune(Module):

    def __init__(self, model_in, data, steps, model_size, model_out):
        self.model_in = model_in # path to model file
        self.model_out = model_out # path to model file
        self.data = data # path to text file
        self.steps = steps # int
        self.model_size = model_size # size of GPT-2 model (e.g., "117M")
        self.ready = checkFiles(data, model_in)
        self.output_files = [model_out]

    def run(self):
        import gpt2
        cwd = os.getcwd()
        # Clear out any existing models that might bein the way
        if os.path.exists(self.model_out):
            if os.path.isdir(self.model_out):
                shutil.rmtree(self.model_out)
            else:
                os.remove(self.model_out)
        # Horrible hack because gpt-2 makes assumptions about relative paths instead of absolute paths
        os.chdir('gpt-2')
        print("Keyboard interrupt will stop training but program will try to continue.")
        try:
            gpt2.train(os.path.join(cwd, self.data), 
                       os.path.join(cwd, self.model_in),
                       os.path.join(cwd, self.model_out), 
                       steps = self.steps, 
                       model_name = self.model_size)
        except KeyboardInterrupt:
            print("keyboard interrupt")
        shutil.copy(os.path.join(cwd, self.model_in, 'encoder.json'), os.path.join(cwd, self.model_out))
        shutil.copy(os.path.join(cwd, self.model_in, 'hparams.json'), os.path.join(cwd, self.model_out))
        shutil.copy(os.path.join(cwd, self.model_in, 'vocab.bpe'), os.path.join(cwd, self.model_out))
        os.chdir('..')

#######################################################

class GPT2_Run(Module):

    def __init__(self, model_in, prompt, model_size, top_k, temperature, num_samples, output):
        self.prompt = prompt # path to prompt text file
        self.model_in = model_in # path to model file
        self.model_size = model_size # size of GPT-2 model (e.g., "117M")
        self.top_k = top_k # int
        self.temperature = temperature # float
        self.num_samples = num_samples # int
        self.output = output # path to output text file
        self.ready = checkFiles(prompt, model_in)
        self.output_files = [output]

    def run(self):
        import gpt2
        cwd = os.getcwd()
        file = open(self.prompt, 'rU')
        prompt_text = file.read()
        prompt_text = prompt_text.strip()
        file.close()
        os.chdir('gpt-2')
        output_text = gpt2.run_gpt(os.path.join(cwd, self.model_in),
                                   model_name = self.model_size,
                                   raw_text = prompt_text,
                                   top_k = self.top_k,
                                   temperature = self.temperature,
                                   nsamples = self.num_samples)
        output_text = prompt_text + output_text
        os.chdir('..')
        with open(self.output, 'w') as outfile:
            outfile.write(output_text)

###############################

'''
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
'''

#####################################

'''
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
'''

#######################################

'''
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
'''

##########################################

class RandomSequence(Module):

    def __init__(self, input, length, output):
        self.input = input     # path to input
        self.output = output   # path to output
        self.length = int(length)   # integer sequence length
        self.ready = checkFiles(input)
        self.output_files = [output]

    def run(self):
        line = '' # final output sequence
        all_lines  = [] # Store all lines in input
        length_lines = [] # store only lines greater than desired length
        for line in open(self.input, 'rU'):
            line = line.strip()
            all_lines.append(line)
            if len(line) > self.length:
                length_lines.append(line)
        # Pick from lines that are long enough if you can
        if len(length_lines) > 0:
            # There are lines long enough
            # Get a line
            pick = random.choice(length_lines)
            # Get a portion of that line
            start = random.randint(0, len(pick)-self.length-1)
            # Picked substring
            line = pick[start:start+self.length]
        else:
            # No lines long enough
            line = random.choice(all_lines)
        # Write the line
        with open(self.output, 'w') as outfile:
            outfile.write(line + NEWLINE)


############################################

class MakeString(Module):

    def __init__(self, string, output):
        self.string = string # A string input
        self.output = output # path to output file
        self.output_files = [output]

    def run(self):
        with open(self.output, 'w') as f:
            f.write(self.string.strip() + NEWLINE)



###############################################

class UserInput(Module):

    def __init__(self, prompt, output):
        self.prompt = prompt # a string with the terminal prompt
        self.output = output # path to output file
        self.output_files = [output]

    def run(self):
        # Default prompt
        prompt = self.prompt.strip()
        if len(prompt) == 0:
            prompt = 'prompt: '
        s = raw_input(prompt)

        with open(self.output, 'w') as f:
            f.write(s.strip() + NEWLINE)

###################################################

'''
class ReadImages(Module):

    def __init__(self, data_directory, output_images):
        self.data_directory = data_directory
        self.output_images = output_images
        if checkFiles(data_directory):
            self.ready = True

    def run(self):
        with open(self.output_images, 'w') as f:
            print >> f, self.data_directory
'''

###################################################

'''
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
'''

###################################################


'''
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
        #   crop = True
        # Get data directory
        f = open(self.input_images, 'r')
        data_dir = f.readline().strip()
        f.close()
        # Create directory for samples and output and checkpoints
        #output_dir = self.output_images + '_output'
        sample_dir = self.model + '_samples'
        checkpoint_dir = CHECKPOINT_DIR
        #if not os.path.exists(output_dir):
        #   os.makedirs(output_dir)
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
        #   print >> f, output_dir

        # Make the training animation
        images = []
        samples = sorted(os.listdir(sample_dir), key=lambda f: os.stat(os.path.join(sample_dir, f)).st_mtime)
        for sample in samples:
            images.append(imageio.imread(os.path.join(sample_dir, sample), 'png'))
        if  len(images) > 0:
            imageio.mimsave(self.animation, images, 'gif')
'''

#####################################################

'''
class DCGAN_Run(Module):

    def __init__ (self, input_images, input_height, output_height, filetype, output_image, model):
        self.input_images = input_images
        self.output_image = output_image
        self.input_height = input_height
        self.output_height = output_height
        self.filetype = filetype
        #self.num_images = num_images
        self.model = model
        self.ready = checkFiles(input_images)

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
        #copyfile(self.model, 'temp/checkpoint')
        gan.run(output_dir = self.output_image, input_height = self.input_height, input_width = self.input_height, output_height = self.output_height, output_width = self.output_height, input_fname_pattern = filetype, dataset = data_dir, checkpoint_dir = 'temp')
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
        self.ready = checkFiles(wiki_directory, input)
        self.output_files = [output]

    def run(self):
        import readWikipedia
        pattern = '' # The pattern is created by the lines of the input file
        first = True # first pattern?
        # Get each line and append it to the pattern
        for line in open(self.input, 'rU'):
            line = line.strip() # line from input file
            if first:
                # First line
                if len(line) > 0:
                    pattern = line
                first = False
            else:
                # Not the first line
                if len(line) > 0:
                    pattern = pattern + '|' + line
        # If a section name is specified, append it to pattern
        if len(self.section_name.strip()) > 0:
            pattern = pattern + ':' + self.section_name.strip()
        readWikipedia.ReadWikipedia(self.wiki_directory, pattern, self.categories, self.output, 'temp/pickfromwikipediatitles')

#######################################

class RandomizeLines(Module):

    def __init__(self, input, output):
        self.input = input    # path to input
        self.output = output  # path to output
        self.ready = checkFiles(input)
        self.output_files = [output]

    def run(self):
        lines = [] # all lines
        # Read the lines
        for line in open(self.input, 'rU'):
            lines.append(line.strip())
        # shuffle the lines
        random.shuffle(lines)
        # write out
        with open(self.output, 'w') as outfile:
            for line in lines:
                outfile.write(line.strip() + NEWLINE)


#########################################

class RemoveTags(Module):

    def __init__(self, input, output):
        self.input = input      # path to input
        self.output = output    # path to output
        self.ready = checkFiles(input)
        self.output_files = [output]

    def run(self):
        lines = [] # all lines
        # Read all lines
        for line in open(self.input, 'rU'):
            # Remove tags from line
            line = re.sub('<[^<]+?>', '', line)
            if len(line.strip()) > 0:
                lines.append(line.strip())
        # Write lines
        with open(self.output, 'w') as outfile:
            for line in lines:
                outfile.write(line.strip() + NEWLINE)                  


##############################

class MakeLowercase(Module):

    def __init__(self, input, output):
        self.input = input
        self.output = output
        self.ready = checkFiles(input)
        self.output_files = [output]

    def run(self):
        with open(self.output, 'w') as outfile:
            for line in open(self.input, 'rU'):
                outfile.write(line.lower().strip() + NEWLINE)

################################

class Wordify(Module):

    def __init__(self, input, output):
        self.input = input    # path to input
        self.output = output  # path to output
        self.ready = checkFiles(input)
        self.output_files = [output]

    def run(self):
        # split at punctuation
        parser = re.compile(r'([\,\.\?\!\'\"\(\)\[\]\{\}])')
        with open(self.output, 'w') as outfile:
            for line in open(self.input, 'rU'):
                line = parser.sub(r' \1 ', line)
                line = re.sub(r'[  ][ ]*', ' ', line) # might put too many spaces in
                outfile.write(line.strip() + NEWLINE)

##################################
### CleanText
### 
### Remove @nbsp; and other html stuff
### Remove @#xxx; characters

class CleanText(Module):

    def __init__(self, input, output):
        self.input = input
        self.output = output
        self.ready = checkFiles(input)
        self.output_files = [output]

    def run(self):
        html_special_characters = {'&nbsp;': ' ',
                                   '&lt;': '<', 
                                   '&gt;': '>', 
                                   '&amp;': '&', 
                                   '&quot;': '"', 
                                   '&apos;': "'", 
                                   '&cent;': '', 
                                   '&pound;': '', 
                                   '&yen;': '',
                                   '&euro;': '',
                                   '&copy;': '',
                                   '&reg;': ''}
        with open(self.output, 'w') as outfile:
            for line in open(self.input, 'rU'):
                # Remove special html characters
                for s in html_special_characters.keys():
                    line = re.sub(s, html_special_characters[s], line)
                    line = re.sub(r'\&\#0-9*;', '', line)
                # remove html tags (probably mainly hyperlinks)
                line = re.sub(r'<[^<]+?>', '', line)
                # remove character name initials and take periods off mr/mrs/ms/dr/etc.
                line = re.sub(r' [M|m]r\.', ' mr', line)
                line = re.sub(r' [M|m]rs\.', ' mrs', line)
                line = re.sub(r' [M|m]s\.', ' ms', line)
                line = re.sub(r' [D|d]r\.', ' dr', line)
                line = re.sub(r' [M|m]d\.', ' md', line)
                line = re.sub(r' [P|p][H|h][D|d]\.', ' phd', line)
                line = re.sub(r' [E|e][S|s][Q|q]\.', ' esq', line)
                line = re.sub(r' [J|d][D|d]\.', ' esq', line)
                line = re.sub(r' [L|l][T|t]\.', ' lt', line)
                line = re.sub(r' [G|g][O|o][V|v]\.', ' lt', line)
                line = re.sub(r' [C|c][P|p][T|t]\.', ' cpt', line)
                line = re.sub(r' [S|s][T|t]\.', ' st', line)
                # handle i.e. and cf.
                line = re.sub(r'i\.e\. ', 'ie ', line)
                line = re.sub(r'cf\. ', 'cf', line)
                # deal with periods in quotes
                line = re.sub(r'\.\"', '\".', line) 
                # remove single letter initials
                p4 = re.compile(r'([ \()])([A-Z|a-z])\.')
                line = p4.sub(r'\1\2', line) 
                # Acroymns with periods are not fun. Need two steps to get rid of those periods.
                # I don't think this is working quite right
                p1 = re.compile(r'([A-Z|a-z])\.([)|\"|\,])')
                line = p1.sub(r'\1\2', line)
                p2 = re.compile(r'\.([A-Z|a-z])')
                line = p2.sub(r'\1', line)
                # periods in numbers
                p3 = re.compile(r'([0-9]+)\.([0-9]+)')
                line = p3.sub(r'\1\2', line)
                outfile.write(line.strip() + NEWLINE)


#####################################

class SaveModel(Module):

    def __init__(self, model, file):
        self.model = model # path to the existing model
        self.file = file   # path to save the model to
        self.ready = checkFiles(model)
        self.output_files = [file]

    def run(self):
        if len(self.file) > 0:
            if os.path.isdir(self.model):
                if os.path.exists(self.file):
                    shutil.rmtree(self.file)
                shutil.copytree(self.model, self.file)
            else:
                shutil.copy(self.model, self.file)


###################################

class SaveDictionary(Module):

    def __init__(self, dictionary, file):
        self.dictionary = dictionary # path to dictionary file
        self.file = file             # path to save to
        self.ready = checkFiles(dictionary)
        self.output_files = [file]

    def run(self):
        if len(self.file) > 0 and os.path.exists(self.dictionary):
            shutil.copy(self.dictionary, self.file)

#####################################

class LoadDictionary(Module):

    def __init__(self, file, dictionary):
        self.dictionary = dictionary # path to save to
        self.file = file             # path to file to load
        self.ready = checkFiles(file)
        self.output_files = [dictionary]

    def run(self):
        if os.path.exists(self.file):
            shutil.copy(self.file, self.dictionary)

######################################

class LoadModel(Module):

    def __init__(self, file, model):
        self.file = file    # path of file to load
        self.model = model  # path to save to
        self.ready = checkFiles(file)
        self.output_files = [model]

    def run(self):
        if os.path.isdir(self.file):
            if os.path.exists(self.model):
                shutil.rmtree(self.model)
            shutil.copytree(self.file, self.model)
        elif os.path.exists(self.file):
            shutil.copyfile(self.file, self.model)


##########################################

class KeepFirstLine(Module):

    def __init__(self, input, output):
        self.input = input       # path of input
        self.output = output     # path of output
        self.ready = checkFiles(input)
        self.output_files = [output]

    def run(self):
        with open(self.output, 'w') as outfile:
            for line in open(self.input, 'rU'):
                outfile.write(line.strip() + NEWLINE)
                break


###########################################

class DeleteFirstLine(Module):

    def __init__(self, input, output):
        self.input = input     # path to input
        self.output = output   # path to output
        self.ready = checkFiles(input)
        self.output_files = [output]

    def run(self):
        with open(self.output, 'w') as outfile:
            first = True # is this the first line?
            for line in open(self.input, 'rU'):
                if first:
                    first = False
                else:
                    outfile.write(line.strip() + NEWLINE)

##########################################

class DeleteLastLine(Module):

    def __init__(self, input, output):
        self.input = input        # path to input
        self.output = output      # path to output
        self.ready = checkFiles(input)
        self.output_files = [output]

    def run(self):
        lines = [] # all lines
        # Read the lines
        for line in open(self.input, 'rU'):
            lines.append(line.strip())
        # Write the lines
        with open(self.output, 'w') as outfile:
            for line in lines[0:len(lines)-1]:
                outfile.write(line.strip() + NEWLINE)

###########################################

class ReadFromWeb(Module):

    def __init__(self, url, data):
        self.url = url
        self.data = data
        self.output_files = [data]
        self.ready = True

    def run(self):
        import urllib.request
        html = ''
        with urllib.request.urlopen(self.url) as response:
            html = response.read()
        with open(self.data, 'w') as outfile:
            outfile.write(html.decode('utf-8', 'ignore').strip() + NEWLINE)

############################################

class ReadAllFromWeb(Module):

    def __init__(self, urls, data):
        self.urls = urls
        self.data = data
        self.ready = checkFiles(urls)
        self.output_files = [data]

    def run(self):
        import urllib.request
        import urllib.error
        html_files = []
        first = True
        for url in open(self.urls, 'r'):
            try:
                if not first:
                    time.sleep(10+random.randint(0,10))
                print('Opening', url)
                with urllib.request.urlopen(url) as response:
                    html_bytes = response.read()
                    html_files.append(html_bytes.decode('utf-8').strip())
            except urllib.error.HTTPError:
                print("Service unavailable")
            first = False
        with open(self.data, 'w') as outfile:
            for html in html_files:
                outfile.write(html.strip() + NEWLINE)

###########################################

class KeepLineWhen(Module):

    def __init__(self, input, match, output):
        self.input = input                     # path to input
        self.output = output                   # path to output
        self.match = convertHexToASCII(match)  # regex pattern to match (with % hex)
        self.ready = checkFiles(input)
        self.output_files = [output]

    def run(self):
        with open(self.output, 'w') as outfile:
            for line in open(self.input, 'rU'):
                m = re.search(self.match, line) # run the regular expression
                if m is not None:
                    outfile.write(line.strip() + NEWLINE)

##########################################

class KeepLineUnless(Module):

    def __init__(self, input, match, output):
        self.input = input     # path to input file
        self.output = output   # path to output
        self.match = convertHexToASCII(match)  #regex pattern
        self.ready = checkFiles(input)
        self.output_files = [output]

    def run(self):
        with open(self.output, 'w') as outfile:
            for line in open(self.input, 'rU'):
                m = re.search(self.match, line) # run the regex
                if m is None:
                    outfile.write(line.strip() + NEWLINE)


#############################################

class MakeCountFile(Module):

    def __init__(self, num, prefix, postfix, output):
        self.num = int(num)      # number of numbers
        self.prefix = prefix     # prefix string
        self.postfix = postfix   # postfix string
        self.output = output     # path to save to
        self.output_files = [output]

    def run(self):
        num = self.num # The max number
        # Check for negative numbers
        if num < 0:
            num = 0
        #Write to file
        with open(self.output, 'w') as outfile:
            for n in range(1, num+1):
                outfile.write(self.prefix.strip() + str(n) + self.postfix.strip() + NEWLINE)

############################################
## This probably doesn't work very well

class SplitHTML(Module):

    def __init__(self, input, output):
        self.input = input                  # path to file with html
        self.output = output                # path to formatted file 
        self.ready = checkFiles(input)
        self.output_files = [output]

    def run(self):
        file = open(self.input, 'rU') # open file
        text = file.read() # read all text
        file.close() # close file
        
        spans = [] # a span containing a tag
        # Final all tags
        for match in re.finditer(r'<[\S\s]+?>', text):
            # Store starts and ends of each tag as a span
            spans.append((match.start(0), match.end(0)))
        # make sure the tags are in order if increasing start
        spans = sorted(spans, key=lambda x: x[0])
        # Add a dummy end span
        spans.append((len(text), len(text)))
        with open(self.output, 'w') as outfile:
            # If no spans, then write the entire text to file
            if len(spans) == 0:
                outfile.write(text)
            pos = 0 # keep track of position in text
            # Iterate through spans
            for span in spans:
                text_span = text[pos:span[0]]    #
                tag_span = text[span[0]:span[1]]
                if len(text_span) > 0:
                    outfile.write(text_span.strip() + NEWLINE)
                if len(tag_span) > 0:
                    outfile.write(tag_span.strip() + NEWLINE)
                pos = span[1]

###########################################

class Regex_Search(Module):

    def __init__(self, input, expression, output, group_1, group_2):
        self.input = input            # path to file to run regex on
        self.expression = expression  # regex expression
        self.output = output          # path to result file - can use parens to specify 2 groups
        self.group_1 = group_1        # path to group output 1
        self.group_2 = group_2        # path to group output 2
        self.ready = checkFiles(input)
        self.output_files = [output, group_1, group_2]

    def run(self):
        file = open(self.input, 'rU') # open input file
        text = file.read() # text from input file
        file.close() # close input file

        with open(self.output, 'w') as outfile:
            with open(self.group_1, 'w') as group1file:
                with open(self.group_2, 'w') as group2file:
                    # Final all matches
                    for match in re.finditer(self.expression, text):
                        num_groups = len(match.groups())
                        outfile.write(match.group(0).strip() + NEWLINE)
                        # If groups found
                        if num_groups > 0:
                            group1file.write(match.group(1).strip() + NEWLINE)
                        else:
                            group1file.write('')
                        if num_groups > 1:
                            group2file.write(match.group(2).strip() + NEWLINE)
                        else:
                            group2file.write('')


#############################################

class Regex_Sub(Module):

    def __init__(self, input, expression, replacement, output):
        self.input = input                    # path to input file 
        self.expression = expression          # regex expression string
        self.replacement = replacement                # replacement string
        self.output = output                  # path to output
        self.ready = checkFiles(input)
        self.output_files = [output]

    def run(self):
        file = open(self.input, 'rU') # open file
        text = file.read() # read all text
        file.close() # close file

        with open(self.output, 'w') as outfile:
            # execute the regex substitution
            new_text = re.subn(self.expression, self.replacement, text)[0]
            outfile.write(new_text)


###################################################

class ReadImageFile(Module):

    def __init__(self, file, output):
        self.file = file
        self.output = output
        self.ready = checkFiles(file)
        self.output_files = [output]

    def run(self):
        copy(self.file, self.output)


###################################################

class WriteImageFile(Module):

    def __init__(self, input, file):
        self.input = input
        self.file = file
        self.ready = checkFiles(input)
        self.output_files = [file]

    def run(self):
        copy(self.input, self.file)


##################################################

'''
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
'''

###################################################

'''
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
'''

##################################          

class Sort(Module):

    def __init__(self, input, output):
        self.input = input      # path to input
        self.output = output    # path to output
        self.ready = checkFiles(input)
        self.output_files = [output]

    def run(self):
        lines = [] # all lines
        # Read lines
        for line in open(self.input, 'rU'):
            lines.append(line.strip())
        # Sort
        sorted_lines = sorted(lines)
        # Write
        with open(self.output, 'w') as outfile:
            for line in sorted_lines:
                outfile.write(line.strip() + NEWLINE)

#####################################

class Reverse(Module):

    def __init__(self, input, output):
        self.input = input                 # path to input
        self.output = output               # path to output
        self.ready = checkFiles(input)
        self.output_files = [output]

    def run(self):
        lines = [] #all lines
        # read lines
        for line in open(self.input, 'rU'):
            lines.append(line)
        # Reverse order destructively
        lines.reverse()
        # write new order out
        with open(self.output, 'w') as outfile:
            for line in lines:
                outfile.write(line.strip())


#####################################

class MakeEmptyText(Module):

    def __init__(self, output):
        self.output = output     # path to output
        self.ready = True
        self.output_files = [output]

    def run(self):
        with open(self.output, 'w') as outfile:
            outfile.write('')


########################################

class PrintText(Module):

    def __init__(self, input):
        self.input = input              # path to input
        self.ready = checkFiles(input)

    def run(self):
        for line in open(self.input, 'rU'):
            print(line.strip())

########################################

class RemoveDuplicates(Module):

    def __init__(self, input, output):
        self.input = input       # path to input file
        self.output = output     # path to output file
        self.ready = checkFiles(input)
        self.output_files = [output]

    def run(self):
        lines = []
        for line in open(self.input, 'r'):
            lines.append(line.strip())
        lines = list(dict.fromkeys(lines))
        with open(self.output, 'w') as outfile:
            for line in lines:
                outfile.write(line.strip() + NEWLINE)

###########################################

class Spellcheck(Module):

    def __init__(self, input, output):
        self.input = input   # path to input file
        self.output = output # path to output file
        self.ready = checkFiles(input)
        self.output_files = [output]

    #def is_possessive(self, word):
    #    return len(word) > 2 and word[-1] == 's' and word[-2] == "'" 

    def run(self):
        from spellchecker import SpellChecker
        spell = SpellChecker()
        lines = []
        for line in open(self.input, 'r'):
            lines.append(line.strip())
        with open(self.output, 'w') as outfile:
            # look at each line
            for line in lines:
                fixed = []
                # look at each word in the line
                for word in line.split():
                    match = re.search(r'([a-zA-Z]*)([\W\w]*)', word)
                    root = match.group(1)
                    rest = match.group(2)
                    # Check if word needs to be stemmed
                    #if self.is_possessive(word):
                    #    # stem word if it's a possessive only
                    #    stemmed = True
                    #    stemmed_word = word[0:-2]
                    misspelled = spell.unknown([root])
                    if len(misspelled) > 0:
                        misspelled = list(misspelled)[0]
                        fix = spell.correction(misspelled)
                        fixed.append(fix + rest)
                    else:
                        fixed.append(word)
                fixed_line = ' '.join(fixed)
                outfile.write(fixed_line + NEWLINE)

### How do I take something line "wintergracefall" and determine that it's cool because it is made
### up of three words mashed together?
### Maybe not a problem because spell check will fail
### I suppose I could find substrings that are also words and see if there is a set of 
### possible substrings that are real words and consume the whole string.
### Would need to be a search because "win" doesn't lead to success but "winter" does.


#####################################

class WebCrawl(Module):

    def __init__(self, url, link_id, link_text, max_hops, output):
        self.url = url              # url of initial visit
        self.link_id = link_id      # if the id of the 'next' link is known
        self.link_text = link_text  # if the link text of the 'next' link is known
        self.max_hops = max_hops    # int: max number of hops
        self.output = output        # path to output file
        self.ready = True
        self.output_files = [output]

    def run(self):
        from selenium import webdriver
        from selenium.common.exceptions import NoSuchElementException
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        wd = webdriver.Chrome('chromedriver',options=options)

        results = []    # store all the html we pull down
        hop_count = 0   # count the hops
        done = False    # are we done?

        # Get the first page
        wd.get("https://en.wiktionary.org/wiki/Category:English_vulgarities")

        while not done and hop_count < self.max_hops:
            # Store html
            print(wd.current_url)
            results.append(wd.page_source)
            link = None                     # The next link
            try:
                if len(self.link_id) > 0 :
                    # Try to find the next link by id
                    link = wd.find_element_by_id(self.link_id)
                elif len(self.link_text) > 0:
                    # Try to find the next link by link text
                    link = wd.find_element_by_partial_link_text(self.link_text)
                else:
                    # Nothing specified
                    done = True
            except NoSuchElementException:
                # couldn't find a link
                done = True
          
            if link is not None:
                # if we have a link, click it
                link.click()
                hop_count = hop_count + 1
            else:
                # No link
                done = True
        # ASSERT: we have at least one result
        with open(self.output, 'w') as outfile:
            for result in results:
                outfile.write(result.strip() + NEWLINE)

