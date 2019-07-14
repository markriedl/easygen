import time
import math
import os
import torch
import codecs
import torch.nn as nn
import torch.optim as optim
import pdb


########################################################
### MODEL
########################################################

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)


#######################################################
### DICTIONARY
#######################################################

class Dictionary(object):
    def __init__(self):
        self.token2idx = {}
        self.idx2token = []

    def add_token(self, word):
        if word not in self.token2idx:
            self.idx2token.append(word)
            self.token2idx[word] = len(self.idx2token) - 1
        return self.token2idx[word]

    def __len__(self):
        return len(self.idx2token)

###############################################################################
### CORPUS
################################################################################

class Corpus(object):

    def __init__(self, train_path):
        self.dictionary = Dictionary()
        self.train_data = self.tokenize(train_path) if train_path is not None and os.path.exists(train_path) else None

    def tokenize(self, path):
        """Tokenizes a text file."""
        pass

class WordCorpus(Corpus):

    def tokenize(self, path):
        super(WordCorpus, self).tokenize(path)
        # Add words to the dictionary
        tokens = 0
        with codecs.open(path, 'r', encoding="utf8") as f:
             for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_token(word)

        # Tokenize file content
        with codecs.open(path, 'r', encoding="utf8") as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.token2idx[word]
                    token += 1
        return ids

class CharCorpus(Corpus):

    def tokenize(self, path):
        super(CharCorpus, self).tokenize(path)
        tokens = 0
        with codecs.open(path, 'r', encoding="utf8") as f:
             for line in f:
                tokens += len(line)
                for c in line:
                    self.dictionary.add_token(c)
        with codecs.open(path, 'r', encoding="utf8") as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                for c in line:
                    ids[token] = self.dictionary.token2idx[c]
                    token += 1
        return ids

###############################################################################
# Globals
###############################################################################

RUN_STEPS = 600
RUN_TEMPERATURE = 0.5
SEED = 1
HISTORY = 35
LAYERS = 2
EPOCHS = 50
HIDDEN_NODES = 512
BATCH_SIZE = 10
MODEL_TYPE = 'GRU'
DROPOUT = 0.2
TIED = False
EMBED_SIZE = HIDDEN_NODES
CLIP = 0.25
LR = 0.0001
LR_DECAY = 0.1
LOG_INTERVAL = 10

#############################################################################
# MAIN entrypoints
############################################################################


def wordLSTM_Run(model_path, dictionary_path, output_path, seed = SEED,
                 steps = RUN_STEPS, temperature = RUN_TEMPERATURE, k = 0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model
    model = None
    with open(model_path, 'rb') as f:
        model = torch.load(f)
        model = model.to(device)
    model.eval()

    # Load the dictionary
    dictionary = None
    with open(dictionary_path, 'rb') as f:
        dictionary = torch.load(f)
    ntokens = len(dictionary)

    hidden = model.init_hidden(1)

    input = torch.randint(ntokens, (1, 1), dtype=torch.long).to(device)

    if seed is not None:
        seed_words = seed.strip().split()
        if len(seed_words) > 1:
            for i in range(len(seed_words)-1):
                word = seed_words[i]
                if word in dictionary.idx2token:
                    input = torch.tensor([[dictionary.token2idx[word]]], dtype=torch.long).to(device)
                    output, hidden = model(input, hidden)
        if len(seed_words) > 0:
            input = torch.tensor([[dictionary.token2idx[seed_words[-1]]]], dtype=torch.long).to(device)

    with codecs.open(output_path, 'w', encoding="utf8") as outf:
        if seed is not None and len(seed) > 0:
            outf.write(seed.strip() + ' ')
        with torch.no_grad():  # no tracking history
            for i in range(steps):
                output, hidden = model(input, hidden)
                word_weights = output.squeeze().div(temperature).exp().cpu()
                word_idx = None
                if k > 0:
                    # top-k sampling
                    word_idx = top_k_sample(word_weights, k)
                else:
                    word_idx = torch.multinomial(word_weights, 1)[0]
                input.fill_(word_idx)
                word = dictionary.idx2token[word_idx]

                outf.write(word + ' ' if i < steps-1 else '')

### Top K sampling
def top_k_sample(logits, k):
    values, _ = torch.topk(logits, k)
    min_value = values.min()
    mask = logits >= min_value
    new_logits = logits * mask.float()
    return torch.multinomial(new_logits, 1)[0]




def charLSTM_Run(model_path, dictionary_path, output_path, seed = SEED,
                 steps = RUN_STEPS, temperature = RUN_TEMPERATURE):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = None
    with open(model_path, 'rb') as f:
        model = torch.load(f)
        model = model.to(device)
    model.eval()

    dictionary = None
    with open(dictionary_path, 'rb') as f:
        dictionary = torch.load(f)
    ntokens = len(dictionary)

    hidden = model.init_hidden(1)

    input = torch.randint(ntokens, (1, 1), dtype=torch.long).to(device)

    if seed is not None:
        if len(seed) > 1:
            for i in range(len(seed)-1):
                char = seed[i]
                if char in dictionary.idx2token:
                    input = torch.tensor([[dictionary.token2idx[char]]], dtype=torch.long).to(device)
                    output, hidden = model(input, hidden)
        if len(seed) > 0:
            input = torch.tensor([[dictionary.token2idx[seed[-1]]]], dtype=torch.long).to(device)


    text = seed
    with torch.no_grad():  # no tracking history
        for i in range(steps):
            output, hidden = model(input, hidden)
            char_weights = output.squeeze().div(temperature).exp().cpu()
            char_idx = torch.multinomial(char_weights, 1)[0]
            input.fill_(char_idx)
            char = dictionary.idx2token[char_idx]
            text = text + char

    with codecs.open(output_path, 'w', encoding="utf8") as outf:
        outf.write(text)

def wordLSTM_Train(train_data_path, 
                   dictionary_path, model_out_path, 
                   history = HISTORY, layers = LAYERS, epochs = EPOCHS, hidden_nodes = HIDDEN_NODES, 
                   batch_size = BATCH_SIZE, model_type=MODEL_TYPE, dropout = DROPOUT, tied = TIED, embed_size = EMBED_SIZE,
                   clip = CLIP, lr = LR, 
                   lr_decay = LR_DECAY, 
                   log_interval = LOG_INTERVAL):
    train(train_data_path, 
               dictionary_path, model_out_path, 
               history = history, layers = layers, epochs = epochs, hidden_nodes = hidden_nodes, 
               batch_size = batch_size, model_type=model_type, dropout = dropout, tied = tied, embed_size = embed_size,
               clip = clip, lr = lr, 
               lr_decay = lr_decay, 
               log_interval = log_interval,
               corpus_type = WordCorpus)

def wordLSTM_Train_More(train_data_path, model_in_path, dictionary_path, model_out_path,
                        history = HISTORY, epochs = EPOCHS, batch_size = BATCH_SIZE, 
                        clip = CLIP, lr = LR, lr_decay = LR_DECAY, log_interval = LOG_INTERVAL):
    train_more(train_data_path, model_in_path, dictionary_path, model_out_path,
               history = history, epochs = epochs, batch_size = batch_size,
               clip = clip, lr = lr, lr_decay = lr_decay, log_interval = log_interval,
               corpus_type = WordCorpus)

def charLSTM_Train(train_data_path, 
                   dictionary_path, model_out_path, 
                   history = HISTORY, layers = LAYERS, epochs = EPOCHS, hidden_nodes = HIDDEN_NODES, 
                   batch_size = BATCH_SIZE, model_type=MODEL_TYPE, dropout = DROPOUT, tied = TIED, embed_size = EMBED_SIZE,
                   clip = CLIP, lr = LR, 
                   lr_decay = LR_DECAY, 
                   log_interval = LOG_INTERVAL):
    train(train_data_path, 
               dictionary_path, model_out_path, 
               history = history, layers = layers, epochs = epochs, hidden_nodes = hidden_nodes, 
               batch_size = batch_size, model_type=model_type, dropout = dropout, tied = tied, embed_size = embed_size,
               clip = clip, lr = lr, 
               lr_decay = lr_decay, 
               log_interval = log_interval,
               corpus_type = CharCorpus)

def charLSTM_Train_More(train_data_path, model_in_path, dictionary_path, model_out_path,
                        history = HISTORY, epochs = EPOCHS, batch_size = BATCH_SIZE, 
                        clip = CLIP, lr = LR, lr_decay = LR_DECAY, log_interval = LOG_INTERVAL):
    train_more(train_data_path, model_in_path, dictionary_path, model_out_path,
               history = history, epochs = epochs, batch_size = batch_size,
               clip = clip, lr = lr, lr_decay = lr_decay, log_interval = log_interval,
               corpus_type = CharCorpus)



#################################################
### TRAIN
##################################################

def train_more(train_data_path, model_in_path, dictionary_path, model_out_path,
               history = HISTORY, epochs = EPOCHS, batch_size = BATCH_SIZE,
               clip = CLIP, lr = LR, lr_decay = LR_DECAY, log_interval = LOG_INTERVAL,
               corpus_type = WordCorpus):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    corpus = corpus_type(train_data_path) 
    dictionary = corpus.dictionary
    with open(dictionary_path, 'wb') as f:
        dictionary = torch.load(f)
    with open(model_in_path, 'wb') as f:
        model = torch.load(f)
        model = model.to(device)
    train_loop(model, corpus, model_out_path, 
               history = history, epochs = epochs, batch_size = batch_size,
               clip = clip, lr = lr, lr_decay = lr_decay, log_interval = log_interval)


def train(train_data_path, dictionary_path, model_out_path, 
               history = HISTORY, layers = LAYERS, epochs = EPOCHS, hidden_nodes = HIDDEN_NODES, 
               batch_size = BATCH_SIZE, model_type=MODEL_TYPE, dropout = DROPOUT, tied = TIED, embed_size = EMBED_SIZE,
               clip = CLIP, lr = LR, 
               lr_decay = LR_DECAY, 
               log_interval = LOG_INTERVAL,
               corpus_type = WordCorpus):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    corpus = corpus_type(train_data_path) 
    dictionary = corpus.dictionary
    with open(dictionary_path, 'wb') as f:
        torch.save(dictionary, f)
    
    ### BUILD THE MODEL
    ntokens = len(dictionary)
    model = RNNModel(model_type, ntokens, embed_size, hidden_nodes, layers, dropout, tied)
    model = model.to(device)
    train_loop(model, corpus, model_out_path, 
               history = history, epochs = epochs, batch_size = batch_size,
               clip = clip, lr = lr, lr_decay = lr_decay, log_interval = log_interval)

def train_loop(model, corpus, model_path,
               history = HISTORY, epochs = EPOCHS, batch_size = BATCH_SIZE, 
               clip = CLIP, lr = LR, lr_decay = LR_DECAY, log_interval = LOG_INTERVAL):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Starting from sequential data, batchify arranges the dataset into columns.
    # For instance, with the alphabet as the sequence and batch size 4, we'd get
    # | a g m s |
    # | b h n t |
    # | c i o u |
    # | d j p v |
    # | e k q w |
    # | f l r x |.
    # These columns are treated as independent by the model, which means that the
    # dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
    # batch processing.
    train_data = batchify(corpus.train_data, batch_size, device)
    val_data = batchify(corpus.train_data[0:corpus.train_data.size()[0]//10], batch_size, device)
    dictionary = corpus.dictionary
    ntokens = len(dictionary)

    criterion = nn.CrossEntropyLoss()

    best_val_loss = None
    log_interval = max(1, (len(train_data) // history) // log_interval)

    ### TRAIN
    for epoch in range(1, epochs+1):
        epoch_start_time = time.time()

        model.train()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        total_loss = 0.0
        start_time = time.time()
        hidden = model.init_hidden(batch_size)

        for batch, i in enumerate(range(0, train_data.size(0) - 1, history)):
            data, targets = get_batch(train_data, i, batch_size)
            # Starting each batch, we detach the hidden state from how it was previously produced.
            # If we didn't, the model would try backpropagating all the way to start of the dataset.
            hidden = repackage_hidden(hidden)
            #model.zero_grad()
            optimizer.zero_grad()
            output, hidden = model(data, hidden)
            loss = criterion(output.view(-1, ntokens), targets)
            loss.backward()
            optimizer.step()

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            '''
            for p in model.parameters():
                p.data.add_(-lr, p.grad.data)
            '''

            total_loss += loss.item()

            if batch % log_interval == 0 and batch > 0:
                cur_loss = total_loss / log_interval
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.5f} | ms/batch {:5.2f} | '
                        'loss {:5.2f} | ppl {:8.2f}'.format(
                        epoch, batch, len(train_data) // history, lr,
                        elapsed * 1000 / log_interval, cur_loss, math.exp(cur_loss) if cur_loss < 1000 else float('inf')))
                total_loss = 0
                start_time = time.time()

        ### EVALUATE
        val_loss = evaluate(model, val_data, criterion, dictionary, batch_size, history)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | train loss {:5.2f} | '
                'train ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                           val_loss, math.exp(val_loss) if val_loss < 1000 else float('inf')))
        print('-' * 89)
        # Save the model if the validation loss is the best we've seen so far.
        if best_val_loss is None:
            best_val_loss = val_loss
        if val_loss <= best_val_loss:
            with open(model_path, 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            lr = lr * lr_decay



#######################################################
### HELPERS
######################################################


def batchify(data, bsz, device):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

# get_batch subdivides the source data into chunks of length args.bptt.
# If source is equal to the example output of the batchify function, with
# a bptt-limit of 2, we'd get the following two Variables for i = 0:
# | a g m s | | b h n t |
# | b h n t | | c i o u |
# Note that despite the name of the function, the subdivison of data is not
# done along the batch dimension (i.e. dimension 1), since that was handled
# by the batchify function. The chunks are along dimension 0, corresponding
# to the seq_len dimension in the LSTM.
def get_batch(source, i, history):
    seq_len = min(history, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target


def evaluate(model, data_source, criterion, dictionary, batch_size, history):
# Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.0
    ntokens = len(dictionary)
    hidden = model.init_hidden(batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, history):
            data, targets = get_batch(data_source, i, batch_size)
            output, hidden = model(data, hidden)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).item()
            hidden = repackage_hidden(hidden)
    return total_loss / (len(data_source) - 1)

######################################
### TESTING



if __name__ == "__main__":
    print("running")
    train_data_path = 'datasets/origin_train'
    val_data_path = 'datasets/origin_valid'
    dictionary_path = 'origin_dictionary'
    model_out_path = 'origin.model'
    output_path = 'foo_out.txt'
    seed = 'the'
    print("training")
    wordLSTM_Train(train_data_path, 
                   dictionary_path, 
                   model_out_path, 
                   epochs = 1)
    print("running")
    wordLSTM_Run(model_out_path, dictionary_path, output_path, seed = seed, k = 20)
