from __future__ import absolute_import, division, print_function

import string
import random
import os
import pickle
from six.moves import urllib

import tensorflow as tf
import tflearn
from tflearn.data_utils import *

'''
pip install -I tensorflow==0.12.1
pip install -I tflearn==0.2.1
'''

# inputs: 
# 	data - textfile
# outputs:
# 	dictionary - char_idx pickle file
# params:
#	history - max length of sequence to feed into neural net
def MakeLSTMDictionary(data, dictionary, history = 25):
	char_idx = None
	X, Y, char_idx = textfile_to_semi_redundant_sequences(data, seq_maxlen=history, redun_step=3)
	pickle.dump(char_idx, open(dictionary,'wb'))
	print(char_idx)
	print(len(char_idx))


# inputs:
#	data - textfile
#	dictionary - char_idx pickle
# outputs:
#	model - a TFlearn model file
# params:
# 	history - max length of sequence to feed into neural net
#	layers - number of hidden layers of the network
# 	epochs - how many epochs to run
#	hidden_nodes - how many nodes per hidden layer
def CharacterLSTM_Train(data, model, dictionary, history = 25, layers = 3, epochs = 10, hidden_nodes = 512, dropout = False):
	char_idx_file = dictionary
	maxlen = history

	char_idx = None
	'''
	if os.path.isfile(char_idx_file):
		print('Loading previous char_idx')
		char_idx = pickle.load(open(char_idx_file, 'rb'))
	print("---------------")
	print(char_idx)
	print(len(char_idx))
	'''

	X, Y, char_idx = textfile_to_semi_redundant_sequences(data, seq_maxlen=maxlen, redun_step=3)

	pickle.dump(char_idx, open(dictionary,'wb'))

	tf.reset_default_graph()
	print("layers " + str(layers) + " hidden " + str(hidden_nodes))
	g = tflearn.input_data([None, maxlen, len(char_idx)])
	for n in range(layers-1):
		g = tflearn.lstm(g, hidden_nodes, return_seq=True)
		if dropout:
			g = tflearn.dropout(g, 0.5)
	g = tflearn.lstm(g, hidden_nodes)
	if dropout:
		g = tflearn.dropout(g, 0.5)
	g = tflearn.fully_connected(g, len(char_idx), activation='softmax')
	g = tflearn.regression(g, optimizer='adam', loss='categorical_crossentropy', learning_rate=0.001)
	'''
	g = tflearn.input_data([None, maxlen, len(char_idx)])
	g = tflearn.lstm(g, 512, return_seq=True)
	g = tflearn.dropout(g, 0.5)
	g = tflearn.lstm(g, 512, return_seq=True)
	g = tflearn.dropout(g, 0.5)
	g = tflearn.lstm(g, 512)
	g = tflearn.dropout(g, 0.5)
	g = tflearn.fully_connected(g, len(char_idx), activation='softmax')
	g = tflearn.regression(g, optimizer='adam', loss='categorical_crossentropy', learning_rate=0.001)
	'''
	m = tflearn.SequenceGenerator(g, dictionary=char_idx, seq_maxlen=maxlen, clip_gradients=5.0) #, checkpoint_path='model_history_gen')

	#if model is not None:
	#	m.load(model)

	#for i in range(epochs):
	seed = random_sequence_from_textfile(data, maxlen)
	m.fit(X, Y, validation_set=0.1, batch_size=128, n_epoch=epochs, run_id='run_gen')
	print("Saving...")
	m.save(model)
	#print("-- TESTING...")
	#print("-- Test with temperature of 1.0 --")
	#print(m.generate(600, temperature=1.0, seq_seed=seed))
	#print("-- Test with temperature of 0.5 --")
	#print(m.generate(600, temperature=0.5, seq_seed=seed))


# inputs:
#	dictionary - char_idx pickle
# 	model - a TFlearn model file
# outputs:
# 	output - a text file
# params:
# 	history - max length of sequence to feed into neural net
#	layers - number of hidden layers of the network
#	hidden_nodes - how many nodes per hidden layer
# 	temperature - float (0..1)
# 	steps - number of characters to generate
# 	seed - a string to kick generation off (preferably from the original data)
def CharacterLSTM_Run(seed, dictionary, model, output, steps = 600, layers = 3, hidden_nodes = 512, history = 25, temperature = 0.5):
	char_idx_file = dictionary
	maxlen = history
	
	char_idx = None
	if os.path.isfile(char_idx_file):
		print('Loading previous char_idx')
		char_idx = pickle.load(open(char_idx_file, 'rb'))

	'''
	g = tflearn.input_data([None, maxlen, len(char_idx)])
	for n in range(layers):
		g = tflearn.lstm(g, hidden_nodes, return_seq=True)
		g = tflearn.dropout(g, 0.5)
	g = tflearn.fully_connected(g, len(char_idx), activation='softmax')
	g = tflearn.regression(g, optimizer='adam', loss='categorical_crossentropy', learning_rate=0.001)
	'''
	tf.reset_default_graph()
	g = tflearn.input_data([None, maxlen, len(char_idx)])
	for n in range(layers-1):
		g = tflearn.lstm(g, hidden_nodes, return_seq=True)
		if dropout:
			g = tflearn.dropout(g, 0.5)
	g = tflearn.lstm(g, hidden_nodes)
	if dropout:
		g = tflearn.dropout(g, 0.5)
	g = tflearn.fully_connected(g, len(char_idx), activation='softmax')
	g = tflearn.regression(g, optimizer='adam', loss='categorical_crossentropy', learning_rate=0.001)


	m = tflearn.SequenceGenerator(g, dictionary=char_idx, seq_maxlen=maxlen, clip_gradients=5.0) #, checkpoint_path='model_history_gen')

	m.load(model)

	#seed = random_sequence_from_textfile(data, maxlen)
	
	print('seed='+seed)
	print('len=' + str(len(seed)))
	result = m.generate(steps, temperature=temperature, seq_seed=seed[:history])
	print (result)
	return result



