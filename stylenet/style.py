from __future__ import print_function
import sys, os, pdb
#sys.path.insert(0, 'src')
import numpy as np, scipy.misc 
from optimize import optimize
from argparse import ArgumentParser
from utils import save_img, get_img, exists, list_files
import evaluate

CONTENT_WEIGHT = 7.5e0
STYLE_WEIGHT = 1e2
TV_WEIGHT = 2e2

LEARNING_RATE = 1e-3
NUM_EPOCHS = 2
CHECKPOINT_DIR = 'checkpoints'
TEST_DIR = CHECKPOINT_DIR
CHECKPOINT_ITERATIONS = 1000
VGG_PATH = 'stylenet/data/imagenet-vgg-verydeep-19.mat'
TRAIN_PATH = 'stylenet/data/train2014'
BATCH_SIZE = 4
DEVICE = '/gpu:0'
FRAC_GPU = 1

def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--checkpoint-dir', type=str,
                        dest='checkpoint_dir', help='dir to save checkpoint in',
                        metavar='CHECKPOINT_DIR', required=True)

    parser.add_argument('--style', type=str,
                        dest='style', help='style image path',
                        metavar='STYLE', required=True)

    parser.add_argument('--train-path', type=str,
                        dest='train_path', help='path to training images folder',
                        metavar='TRAIN_PATH', default=TRAIN_PATH)

    parser.add_argument('--test', type=str,
                        dest='test', help='test image path',
                        metavar='TEST', default=False)

    parser.add_argument('--test-dir', type=str,
                        dest='test_dir', help='test image save dir',
                        metavar='TEST_DIR', default=False)

    parser.add_argument('--slow', dest='slow', action='store_true',
                        help='gatys\' approach (for debugging, not supported)',
                        default=False)

    parser.add_argument('--epochs', type=int,
                        dest='epochs', help='num epochs',
                        metavar='EPOCHS', default=NUM_EPOCHS)

    parser.add_argument('--batch-size', type=int,
                        dest='batch_size', help='batch size',
                        metavar='BATCH_SIZE', default=BATCH_SIZE)

    parser.add_argument('--checkpoint-iterations', type=int,
                        dest='checkpoint_iterations', help='checkpoint frequency',
                        metavar='CHECKPOINT_ITERATIONS',
                        default=CHECKPOINT_ITERATIONS)

    parser.add_argument('--vgg-path', type=str,
                        dest='vgg_path',
                        help='path to VGG19 network (default %(default)s)',
                        metavar='VGG_PATH', default=VGG_PATH)

    parser.add_argument('--content-weight', type=float,
                        dest='content_weight',
                        help='content weight (default %(default)s)',
                        metavar='CONTENT_WEIGHT', default=CONTENT_WEIGHT)
    
    parser.add_argument('--style-weight', type=float,
                        dest='style_weight',
                        help='style weight (default %(default)s)',
                        metavar='STYLE_WEIGHT', default=STYLE_WEIGHT)

    parser.add_argument('--tv-weight', type=float,
                        dest='tv_weight',
                        help='total variation regularization weight (default %(default)s)',
                        metavar='TV_WEIGHT', default=TV_WEIGHT)
    
    parser.add_argument('--learning-rate', type=float,
                        dest='learning_rate',
                        help='learning rate (default %(default)s)',
                        metavar='LEARNING_RATE', default=LEARNING_RATE)

    return parser

def check_opts(opts):
    exists(opts.checkpoint_dir, "checkpoint dir not found!")
    exists(opts.style, "style path not found!")
    exists(opts.train_path, "train path not found!")
    if opts.test or opts.test_dir:
        exists(opts.test, "test img not found!")
        exists(opts.test_dir, "test directory not found!")
    exists(opts.vgg_path, "vgg network data not found!")
    assert opts.epochs > 0
    assert opts.batch_size > 0
    assert opts.checkpoint_iterations > 0
    assert os.path.exists(opts.vgg_path)
    assert opts.content_weight >= 0
    assert opts.style_weight >= 0
    assert opts.tv_weight >= 0
    assert opts.learning_rate >= 0

def _get_files(img_dir):
    files = list_files(img_dir)
    return [os.path.join(img_dir,x) for x in files]

    
def main(style, test = False, test_dir = 'test', train_path = TRAIN_PATH, slow = False, epochs = NUM_EPOCHS, checkpoint_iterations = CHECKPOINT_ITERATIONS, batch_size = BATCH_SIZE, checkpoint_dir = CHECKPOINT_DIR, learning_rate = LEARNING_RATE, content_weight = CONTENT_WEIGHT, style_weight = STYLE_WEIGHT, tv_weight = TV_WEIGHT, vgg_path = VGG_PATH):
    #parser = build_parser()
    #options = parser.parse_args()
    #check_opts(options)

    style_target = get_img(style)
    if not slow:
        content_targets = _get_files(train_path)
    elif test:
        content_targets = [test]

    kwargs = {
        "slow":slow,
        "epochs":epochs,
        "print_iterations":checkpoint_iterations,
        "batch_size":batch_size,
        "save_path":checkpoint_dir,
        "learning_rate":learning_rate
    }

    if slow:
        if epochs < 10:
            kwargs['epochs'] = 1000
        if learning_rate < 1:
            kwargs['learning_rate'] = 1e1

    args = [
        content_targets,
        style_target,
        content_weight,
        style_weight,
        tv_weight,
        vgg_path
    ]

    for preds, losses, i, epoch in optimize(*args, **kwargs):
        style_loss, content_loss, tv_loss, loss = losses

        print('Epoch %d, Iteration: %d, Loss: %s' % (epoch, i, loss))
        to_print = (style_loss, content_loss, tv_loss)
        print('style: %s, content:%s, tv: %s' % to_print)
        if test:
            assert test_dir != False
            preds_path = '%s/%s_%s.png' % (test_dir,epoch,i)
            if not slow:
                ckpt_dir = os.path.dirname(checkpoint_dir)
                evaluate.ffwd_to_img(test,preds_path,
                                     checkpoint_dir)
            else:
                save_img(preds_path, img)
    #ckpt_dir = checkpoint_dir
    #cmd_text = 'python evaluate.py --checkpoint-dir %s ...' % ckpt_dir
    #print("Training complete. For evaluation:\n    `%s`" % cmd_text)

if __name__ == '__main__':
    main()
