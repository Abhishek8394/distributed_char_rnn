from __future__ import print_function
import argparse
import os
import codecs
from utils import create_vocab_file
import numpy as np

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--data_dir", help="Data directory containing the dataset.", required=True)
	parser.add_argument("--num_parts", help="Number of parts in which to divide, same as amount of worker nodes you plan to have.", type=int, required=True)
	parser.add_argument("--out_dir", help="Output directory. Defaults to data_dir. Will contain files as  'data-<num>.npy'", default=None)

	args = parser.parse_args()
	inp_file = os.path.join(args.data_dir, "input.txt")
	vocab_file = os.path.join(args.data_dir, "vocab.pkl")
	out_dir = args.data_dir if args.out_dir is None else args.out_dir
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)
	base_out_file = os.path.join(out_dir, "data")
	# create a vocabulary on whole dataset. This is common for all nodes.
	print("building vocabulary...")
	res = create_vocab_file(inp_file, vocab_file)
	vocab = res["vocab"]
	data = res["data"]
	# begin sharding
	print("sharding file...")
	
	tensor = np.array(list(map(vocab.get, data)))
	# split in equal parts.
	split_tensors = np.split(tensor, args.num_parts)
	for i, tensor in enumerate(split_tensors):
		print("writing shard %d.."%i, end='\r')
		out_file = base_out_file + ("-%d.npy" % i)
		np.save(out_file, tensor)
	print("\ndone")


