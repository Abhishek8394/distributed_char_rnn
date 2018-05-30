Distributed Character RNN
===
This is a port for running a character rnn with distributed tensorflow.
Based on the original code from [https://github.com/sherjilozair/char-rnn-tensorflow](https://github.com/sherjilozair/char-rnn-tensorflow)

## Requirements
- [~Tensorflow 1.4~ Tensorflow 1.8](http://www.tensorflow.org)

## Basic Usage

#### 1. For launching a distributed training environment

**Step 1:** First make sure sure your data is sharde if you want data parallel training. For creating that run the following command:

```bash
# Call with -h option for more help
python data_splitter.py --data_dir data/tinyshakespeare --num_parts 2 --out_dir sharded_data
```

**Step 2:** You need to launch each node as a different process. The command for launching any node is 

```bash
 python train.py --distributed --ps_hosts 127.0.0.1:8000 --worker_hosts 127.0.0.1:9000,127.0.0.1:9001 --job_name $job_name --task_index $task_index --save_dir distrib-train
```

OR, execute the file [launch.bat](https://github.com/Abhishek8394/distributed_char_rnn/blob/master/launch.bat) or [launch.sh](https://github.com/Abhishek8394/distributed_char_rnn/blob/master/launch.sh) to quickly launch a distributed experiment with default settings. 

The options `--job_name` takes value either **ps** or **worker** based on the node's role. Refer to this [TF tutorial](https://www.tensorflow.org/deploy/distributed#specifying_distributed_devices_in_your_model) for more info on these roles.
Similarly `--task_index` takes an integer indicating which node it is. **i**th worker node takes value **i**.

For more options run `python train.py --help`. Note any options you set must be same across all nodes except for node dependent settings like *job_name, task_index*, etc.

#### 2. For running on a single process; without the distributed mode 

To train with default parameters on the tinyshakespeare corpus, run `python train.py`. To access all the parameters use `python train.py --help`.

To sample from a checkpointed model, `python sample.py`.
Sampling while the learning is still in progress (to check last checkpoint) works only in CPU or using another GPU.
To force CPU mode, use `export CUDA_VISIBLE_DEVICES=""` and `unset CUDA_VISIBLE_DEVICES` afterward
(resp. `set CUDA_VISIBLE_DEVICES=""` and `set CUDA_VISIBLE_DEVICES=` on Windows).

To continue training after interruption or to run on more epochs, `python train.py --init_from=save`

## Datasets
You can use any plain text file as input. For example you could download [The complete Sherlock Holmes](https://sherlock-holm.es/ascii/) as such:

```bash
cd data
mkdir sherlock
cd sherlock
wget https://sherlock-holm.es/stories/plain-text/cnus.txt
mv cnus.txt input.txt
```

Then start train from the top level directory using `python train.py --data_dir=./data/sherlock/`

A quick tip to concatenate many small disparate `.txt` files into one large training file: `ls *.txt | xargs -L 1 cat >> input.txt`.

## Tensorboard
To visualize training progress, model graphs, and internal state histograms:  fire up Tensorboard and point it at your `log_dir`.  E.g.:
```bash
$ tensorboard --logdir=./logs/
```

Then open a browser to [http://localhost:6006](http://localhost:6006) or the correct IP/Port specified.

## Contributing
Feel free to send pull requests. Especially related to simplifying the setup as much as possible.
