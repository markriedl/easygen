import json
import os
import numpy as np
import tensorflow as tf
from tensorflow.core.protobuf import rewriter_config_pb2
import sys
import time
import pdb
import shutil

CUR_PATH = os.getcwd()
GPT_PATH = os.path.join(CUR_PATH, 'gpt-2')
sys.path.insert(0, os.path.join(GPT_PATH, 'src'))


import model
import sample
import encoder
import load_dataset
from load_dataset import load_dataset, Sampler
from accumulate import AccumulatingOptimizer
import memory_saving_gradients



####################################################

def run_gpt(
    model_in_path,
    model_name='117M',
    raw_text = ' ',
    seed=None,
    nsamples=1,
    batch_size=1,
    length=None,
    temperature=1,
    top_k=0,
    top_p=0.0
):
    """
    Interactively run the model
    :model_name=117M : String, which model to use
    :seed=None : Integer seed for random number generators, fix seed to reproduce
     results
    :nsamples=1 : Number of samples to return total
    :batch_size=1 : Number of batches (only affects speed/memory).  Must divide nsamples.
    :length=None : Number of tokens in generated text, if None (default), is
     determined by model hyperparameters
    :temperature=1 : Float value controlling randomness in boltzmann
     distribution. Lower temperature results in less random completions. As the
     temperature approaches zero, the model will become deterministic and
     repetitive. Higher temperature results in more random completions.
    :top_k=0 : Integer value controlling diversity. 1 means only 1 word is
     considered for each step (token), resulting in deterministic completions,
     while 40 means 40 words are considered at each step. 0 (default) is a
     special setting meaning no restrictions. 40 generally is a good value.
    :top_p=0.0 : Float value controlling diversity. Implements nucleus sampling,
     overriding top_k if set to a value > 0. A good setting is 0.9.
    """
    output_text = ''

    if batch_size is None:
        batch_size = 1
    assert nsamples % batch_size == 0

    enc = get_encoder(model_in_path)
    hparams = model.default_hparams()
    with open(os.path.join(model_in_path, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    if length is None:
        length = hparams.n_ctx // 2
    elif length > hparams.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)

    with tf.Session(graph=tf.Graph()) as sess:
        context = tf.placeholder(tf.int32, [batch_size, None])
        np.random.seed(seed)
        tf.set_random_seed(seed)
        output = sample.sample_sequence(
            hparams=hparams, length=length,
            context=context,
            batch_size=batch_size,
            temperature=temperature, top_k=top_k, top_p=top_p
        )

        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(model_in_path) #os.path.join('models', model_name))
        saver.restore(sess, ckpt)

        if len(raw_text) == 0:
            raw_text = ' '
        context_tokens = enc.encode(raw_text)
        generated = 0
        for n in range(nsamples // batch_size):
            out = sess.run(output, feed_dict={
                context: [context_tokens for _ in range(batch_size)]
            })[:, len(context_tokens):]
            for i in range(batch_size):
                text = enc.decode(out[i])
                generated = generated + 1
                #print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)
                #print(text)
                if n == 0:
                    output_text = text
                else:
                    output_text = output_text + '\n' + text
    return output_text

##############################################################################

def maketree(path):
    try:
        os.makedirs(path)
    except:
        pass


def randomize(context, hparams, p):
    if p > 0:
        mask = tf.random.uniform(shape=tf.shape(context)) < p
        noise = tf.random.uniform(shape=tf.shape(context), minval=0, maxval=hparams.n_vocab, dtype=tf.int32)
        return tf.where(mask, noise, context)
    else:
        return context


def get_encoder(path):
    with open(os.path.join(path, 'encoder.json'), 'r') as f:
        enc = json.load(f)
    with open(os.path.join(path, 'vocab.bpe'), 'r', encoding="utf-8") as f:
        bpe_data = f.read()
    bpe_merges = [tuple(merge_str.split()) for merge_str in bpe_data.split('\n')[1:-1]]
    return encoder.Encoder(
        encoder=enc,
        bpe_merges=bpe_merges,
    )

def train(dataset, model_in_path, model_out_path,
          model_name = '117M', 
          steps = 1000,
          combine = 50000, 
          batch_size = 1, 
          learning_rate = 0.00002,
          accumulate_gradients = 1,
          memory_saving_gradients = False,
          only_train_transformer_layers = False,
          optimizer = 'adam',
          noise = 0.0,
          top_k = 40,
          top_p = 0.0,
          restore_from = 'latest',
          sample_every = 100,
          sample_length = 1023,
          sample_num = 1,
          save_every = 1000,
          val_dataset = None):
    # Reset the TF computation graph
    tf.reset_default_graph()
    # Get the checkpoint and sample directories
    #checkpoint_dir = os.path.dirname(model_path)
    #sample_dir = checkpoint_dir
    #run_name = os.path.basename(model_path)
    # Load the encoder
    enc = get_encoder(model_in_path)
    hparams = model.default_hparams()
    with open(os.path.join(model_in_path, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    if sample_length > hparams.n_ctx:
        raise ValueError(
            "Can't get samples longer than window size: %s" % hparams.n_ctx)
    
    # Size matters
    if model_name == '345M':
        memory_saving_gradients = True
        if optimizer == 'adam':
            only_train_transformer_layers = True

    # Configure TF
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.graph_options.rewrite_options.layout_optimizer = rewriter_config_pb2.RewriterConfig.OFF
    # Start the session
    with tf.Session(config=config) as sess:
        context = tf.placeholder(tf.int32, [batch_size, None])
        context_in = randomize(context, hparams, noise)
        output = model.model(hparams=hparams, X=context_in)
        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=context[:, 1:], logits=output['logits'][:, :-1]))

        tf_sample = sample.sample_sequence(
            hparams=hparams,
            length=sample_length,
            context=context,
            batch_size=batch_size,
            temperature=1.0,
            top_k=top_k,
            top_p=top_p)

        all_vars = [v for v in tf.trainable_variables() if 'model' in v.name]
        train_vars = [v for v in all_vars if '/h' in v.name] if only_train_transformer_layers else all_vars

        if optimizer == 'adam':
            opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
        elif optimizer == 'sgd':
            opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        else:
            exit('Bad optimizer:', optimizer)

        if accumulate_gradients > 1:
            if memory_saving_gradients:
                exit("Memory saving gradients are not implemented for gradient accumulation yet.")
            opt = AccumulatingOptimizer(
                opt=opt,
                var_list=train_vars)
            opt_reset = opt.reset()
            opt_compute = opt.compute_gradients(loss)
            opt_apply = opt.apply_gradients()
            summary_loss = tf.summary.scalar('loss', opt_apply)
        else:
            if memory_saving_gradients:
                opt_grads = memory_saving_gradients.gradients(loss, train_vars)
            else:
                opt_grads = tf.gradients(loss, train_vars)
            opt_grads = list(zip(opt_grads, train_vars))
            opt_apply = opt.apply_gradients(opt_grads)
            summary_loss = tf.summary.scalar('loss', loss)

        summary_lr = tf.summary.scalar('learning_rate', learning_rate)
        summaries = tf.summary.merge([summary_lr, summary_loss])

        summary_log = tf.summary.FileWriter(
            #os.path.join(checkpoint_dir, run_name)
            model_out_path
            )

        saver = tf.train.Saver(
            var_list=all_vars,
            max_to_keep=1)
        sess.run(tf.global_variables_initializer())

        if restore_from == 'latest':
            ckpt = tf.train.latest_checkpoint(
                #os.path.join(checkpoint_dir, run_name)
                model_in_path
                )
            if ckpt is None:
                # Get fresh GPT weights if new run.
                ckpt = tf.train.latest_checkpoint(
                    model_in_path)#os.path.join('models', model_name))
        elif restore_from == 'fresh':
            ckpt = tf.train.latest_checkpoint(
                model_in_path)#os.path.join('models', model_name))
        else:
            ckpt = tf.train.latest_checkpoint(restore_from)
        print('Loading checkpoint', ckpt)
        saver.restore(sess, ckpt)

        print('Loading dataset...')
        chunks = load_dataset(enc, dataset, combine)
        data_sampler = Sampler(chunks)
        print('dataset has', data_sampler.total_size, 'tokens')
        print('Training...')
        counter = 1
        counter_path = os.path.join(model_in_path, 'counter') #os.path.join(checkpoint_dir, run_name, 'counter')
        if restore_from == 'latest' and os.path.exists(counter_path):
            # Load the step number if we're resuming a run
            # Add 1 so we don't immediately try to save again
            with open(counter_path, 'r') as fp:
                counter = int(fp.read()) + 1

        def save():
            #maketree(os.path.join(checkpoint_dir, run_name))
            maketree(model_out_path)
            print(
                'Saving',
                #os.path.join(checkpoint_dir, run_name, 'model-{}').format(counter)
                os.path.join(model_out_path, 'model-{}').format(counter)
                )
            saver.save(
                sess,
                #os.path.join(checkpoint_dir, run_name, 'model'),
                os.path.join(model_out_path, 'model'),
                global_step=counter)
            with open(os.path.join(model_out_path, 'counter'), 'w') as fp:
                fp.write(str(counter) + '\n')
        
        def generate_samples():
            print('Generating samples...')
            context_tokens = data_sampler.sample(1)
            all_text = []
            index = 0
            while index < sample_num:
                out = sess.run(
                    tf_sample,
                    feed_dict={context: batch_size * [context_tokens]})
                for i in range(min(sample_num - index, batch_size)):
                    text = enc.decode(out[i])
                    text = '======== SAMPLE {} ========\n{}\n'.format(
                        index + 1, text)
                    all_text.append(text)
                    index += 1
            print(text)
            #maketree(os.path.join(sample_dir, run_name))
            maketree(model_out_path)
            with open(os.path.join(model_out_path, 'samples-{}').format(counter), 'w') as fp:
                fp.write('\n'.join(all_text))
        
        def sample_batch():
            return [data_sampler.sample(1024) for _ in range(batch_size)]


        avg_loss = (0.0, 0.0)
        start_time = time.time()

        stop = steps + counter

        try:
            while counter < stop + 1:
                if counter % save_every == 0:
                    save()
                '''
                if counter % sample_every == 0:
                    generate_samples()
                '''

                if accumulate_gradients > 1:
                    sess.run(opt_reset)
                    for _ in range(accumulate_gradients):
                        sess.run(
                            opt_compute, feed_dict={context: sample_batch()})
                    (v_loss, v_summary) = sess.run((opt_apply, summaries))
                else:
                    (_, v_loss, v_summary) = sess.run(
                        (opt_apply, loss, summaries),
                        feed_dict={context: sample_batch()})

                summary_log.add_summary(v_summary, counter)

                avg_loss = (avg_loss[0] * 0.99 + v_loss,
                            avg_loss[1] * 0.99 + 1.0)

                print(
                    '[{counter} | {time:2.2f}] loss={loss:2.2f} avg={avg:2.2f}'
                    .format(
                        counter=counter,
                        time=time.time() - start_time,
                        loss=v_loss,
                        avg=avg_loss[0] / avg_loss[1]))

                counter += 1
            print('done!')
            save()
        except KeyboardInterrupt:
            print('interrupted')
            save()

#####################################################################

if __name__ == '__main__':
    cache_path = os.path.join(os.getcwd(), 'cache')
    os.chdir('gpt-2')
    train(os.path.join(cache_path, 'text0'), cache_path, cache_path, steps=1)
    #text = run_gpt(top_k=40)
    #print(text)
    os.chdir('..')