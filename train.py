#!/usr/bin/env python

from __future__ import print_function, division

import argparse
import time
import os
from six.moves import cPickle

import tensorflow as tf
from utils import TextLoader
from model import Model


parser = argparse.ArgumentParser(
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# Data and model checkpoints directories
parser.add_argument('--data_dir', type=str, default='data/tinyshakespeare',
                    help='data directory containing input.txt with training examples')
parser.add_argument('--save_dir', type=str, default='save',
                    help='directory to store checkpointed models')
parser.add_argument('--log_dir', type=str, default='logs',
                    help='directory to store tensorboard logs')
parser.add_argument('--save_every', type=int, default=1000,
                    help='Save frequency. Number of passes between checkpoints of the model.')
parser.add_argument('--init_from', type=str, default=None,
                    help="""continue training from saved model at this path (usually "save").
                        Path must contain files saved by previous training process:
                        'config.pkl'        : configuration;
                        'chars_vocab.pkl'   : vocabulary definitions;
                        'checkpoint'        : paths to model file(s) (created by tf).
                                              Note: this file contains absolute paths, be careful when moving files around;
                        'model.ckpt-*'      : file(s) with model definition (created by tf)
                         Model params must be the same between multiple runs (model, rnn_size, num_layers and seq_length).
                    """)
# Model params
parser.add_argument('--model', type=str, default='lstm',
                    help='lstm, rnn, gru, or nas')
parser.add_argument('--rnn_size', type=int, default=128,
                    help='size of RNN hidden state')
parser.add_argument('--num_layers', type=int, default=2,
                    help='number of layers in the RNN')
# Optimization
parser.add_argument('--seq_length', type=int, default=50,
                    help='RNN sequence length. Number of timesteps to unroll for.')
parser.add_argument('--batch_size', type=int, default=50,
                    help="""minibatch size. Number of sequences propagated through the network in parallel.
                            Pick batch-sizes to fully leverage the GPU (e.g. until the memory is filled up)
                            commonly in the range 10-500.""")
parser.add_argument('--num_epochs', type=int, default=50,
                    help='number of epochs. Number of full passes through the training examples.')
parser.add_argument('--grad_clip', type=float, default=5.,
                    help='clip gradients at this value')
parser.add_argument('--learning_rate', type=float, default=0.002,
                    help='learning rate')
parser.add_argument('--decay_rate', type=float, default=0.97,
                    help='decay rate for rmsprop')
parser.add_argument('--output_keep_prob', type=float, default=1.0,
                    help='probability of keeping weights in the hidden layer')
parser.add_argument('--input_keep_prob', type=float, default=1.0,
                    help='probability of keeping weights in the input layer')

# distributed args
parser.add_argument('--distributed', help="Indicates running in distributed mode", action='store_true')
parser.add_argument('--ps_hosts', help="PS HOSTS", default=None)
parser.add_argument('--worker_hosts', help="WORKER HOSTS", default=None)
parser.add_argument('--job_name', help="Job name. Must be ps/worker", choices=['ps', 'worker'], default=None)
parser.add_argument('--task_index', help="Index of task for given job", type=int, default=None)

args = parser.parse_args()

def train(args):
    data_loader = TextLoader(args.data_dir, args.batch_size, args.seq_length)
    args.vocab_size = data_loader.vocab_size

    # check compatibility if training is continued from previously saved model
    if args.init_from is not None:
        # check if all necessary files exist
        assert os.path.isdir(args.init_from)," %s must be a a path" % args.init_from
        assert os.path.isfile(os.path.join(args.init_from,"config.pkl")),"config.pkl file does not exist in path %s"%args.init_from
        assert os.path.isfile(os.path.join(args.init_from,"chars_vocab.pkl")),"chars_vocab.pkl.pkl file does not exist in path %s" % args.init_from
        ckpt = tf.train.latest_checkpoint(args.init_from)
        assert ckpt, "No checkpoint found"

        # open old config and check if models are compatible
        with open(os.path.join(args.init_from, 'config.pkl'), 'rb') as f:
            saved_model_args = cPickle.load(f)
        need_be_same = ["model", "rnn_size", "num_layers", "seq_length"]
        for checkme in need_be_same:
            assert vars(saved_model_args)[checkme]==vars(args)[checkme],"Command line argument and saved model disagree on '%s' "%checkme

        # open saved vocab/dict and check if vocabs/dicts are compatible
        with open(os.path.join(args.init_from, 'chars_vocab.pkl'), 'rb') as f:
            saved_chars, saved_vocab = cPickle.load(f)
        assert saved_chars==data_loader.chars, "Data and loaded model disagree on character set!"
        assert saved_vocab==data_loader.vocab, "Data and loaded model disagree on dictionary mappings!"

    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)
    with open(os.path.join(args.save_dir, 'config.pkl'), 'wb') as f:
        cPickle.dump(args, f)
    with open(os.path.join(args.save_dir, 'chars_vocab.pkl'), 'wb') as f:
        cPickle.dump((data_loader.chars, data_loader.vocab), f)

    if args.distributed:
        # get the list of ps and worker hosts
        ps_hosts = args.ps_hosts.split(",")
        worker_hosts = args.worker_hosts.split(",")
        job_name, task_index = args.job_name, args.task_index # get current job and index.
        # hide GPU from ps nodes, they dont need it.
        if job_name == "ps":
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
        # create a cluster.
        cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
        # create a tensorflow server
        server = tf.train.Server(cluster, job_name=job_name, task_index=task_index)
        if job_name == "ps":
            server.join()
        # define our graph
        with tf.device(tf.train.replica_device_setter(cluster=cluster)):
            model = Model(args)
    else:
        model = Model(args)

    # instrument for tensorboard, must be defined before monitored session
    # If your model defined its own tf.Graph object then the
    # following ops must be created in same graph.
    summaries = tf.summary.merge_all()
    writer = tf.summary.FileWriter(
            os.path.join(args.log_dir, time.strftime("%Y-%m-%d-%H-%M-%S")))
    writer.add_graph(tf.get_default_graph())
    # dont need this in distributed training scenario.
    saver = tf.train.Saver(tf.global_variables())
    # op to decay the learning rate
    tf_epoch = tf.placeholder(tf.float32, shape=(), name="epoch_number")
    learning_rate_modifier = tf.assign(model.lr, tf.constant(args.learning_rate) * (tf.constant(args.decay_rate) ** tf_epoch))

    # If you're on a multiple gpu system, set the gpu environment vairable before launching
    # and maybe remove the `per_process_memory_fraction`
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=1/cluster.num_tasks("worker")))

    if args.distributed:
        # create a training session
        # ALERT: No more ops can be defined after this point. 
        mon_sess = tf.train.MonitoredTrainingSession(master=server.target,
                                                     is_chief=(task_index==0),
                                                     checkpoint_dir=args.save_dir,
                                                     chief_only_hooks=[],
                                                     save_checkpoint_steps=args.save_every,
                                                     save_summaries_steps=100
                                                     )
        """We can set checkpoint_{secs,steps} and summaries_{secs,steps} to None
           if we want to manually handle those. But that is not the recommended way
           since in principle only master should be in charge of those operations.
        """
        # This is needed if running ops other than the main training step or ones that dont 
        # increment the global step. Eg. accuracy calculation. This is because running mon_sess
        # might trigger the save summary step and that would fail since there are no related operations
        # being ran in the current call. 
        # Rule of thumb: If it is linked to a op/summary op, that does NOT require training data feed, 
        # it goes with normal_sess 
        normal_sess = mon_sess._tf_sess()
        # note that we dont need to call global_vairable_initializer anymore, monitored session does that for us.
        # same goes for restoring models. It will automatically do it if possible.

    else:
        mon_sess = normal_sess = tf.Session()
        normal_sess.run(tf.global_variables_initializer())
        # restore model
        if args.init_from is not None:
            saver.restore(sess, ckpt)

    for e in range(args.num_epochs):
        # this op goes in normal_sess since it is not a training step.
        normal_sess.run(learning_rate_modifier, feed_dict={tf_epoch: e})
        data_loader.reset_batch_pointer()
        # again, not a training op. 
        state = normal_sess.run(model.initial_state)
        for b in range(data_loader.num_batches):
            start = time.time()
            x, y = data_loader.next_batch()
            feed = {model.input_data: x, model.targets: y}
            for i, (c, h) in enumerate(model.initial_state):
                feed[c] = state[i].c
                feed[h] = state[i].h
            
            summ, train_loss, state, _ = mon_sess.run([summaries, model.cost, model.final_state, model.train_op], feed)
            if not args.distributed:
                # do this only in single mode, distributed takes care of this.
                writer.add_summary(summ, e * data_loader.num_batches + b)

            end = time.time()
            print("{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}"
                  .format(normal_sess.run(model.global_step), #e * data_loader.num_batches + b,
                          args.num_epochs * data_loader.num_batches,
                          e, train_loss, end - start))
            # save stuff manually only if NOT in distributed mode.
            if not args.distributed and ((e * data_loader.num_batches + b) % args.save_every == 0\
                                or (e == args.num_epochs-1 and
                                    b == data_loader.num_batches-1)):
                # save for the last result                
                checkpoint_path = os.path.join(args.save_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path,
                           global_step=e * data_loader.num_batches + b)
                print("model saved to {}".format(checkpoint_path))
    mon_sess.close()


if __name__ == '__main__':
    train(args)
