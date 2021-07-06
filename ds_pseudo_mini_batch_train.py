import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dgl.nn.pytorch as dglnn
import time
import argparse
import tqdm
import deepspeed
import random
# from DeepSpeedEngine_GNN import DeepSpeedEngine_GNN
from utils import Timers
from load_graph import load_reddit, inductive_split, load_ogb
from memory_usage import see_memory_usage
import tracemalloc
from cpu_mem_usage import get_memory

def set_seed(args):
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	if args.local_rank >= 0:
		torch.cuda.manual_seed_all(args.seed)
		torch.cuda.manual_seed(args.seed)
		torch.backends.cudnn.enabled = False
		torch.backends.cudnn.deterministic = True


def ttt(tic, str1):
	toc = time.time()
	print(str1 + ' step Time(s): {:.4f}'.format(toc - tic))
	return toc


# aggre = 'lstm'
# aggre = 'mean'


class SAGE(nn.Module):
	def __init__(self,
				 in_feats,
				 n_hidden,
				 n_classes,
				 n_layers,
				 activation,
				 dropout,
	             aggre):
		super().__init__()
		self.n_layers = n_layers
		self.n_hidden = n_hidden
		self.n_classes = n_classes
		self.layers = nn.ModuleList()
		self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, aggre))
		for i in range(1, n_layers - 1):
			self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, aggre))
		self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, aggre))
		self.dropout = nn.Dropout(dropout)
		self.activation = activation

	def forward(self, blocks, x):
		h = x
		for l, (layer, block) in enumerate(zip(self.layers, blocks)):
			h = layer(block, h)
			if l != len(self.layers) - 1:
				h = self.activation(h)
				h = self.dropout(h)
		return h

	def inference(self, g, x, device):
		"""
		Inference with the GraphSAGE model on full neighbors (i.e. without neighbor sampling).
		g : the entire graph.
		x : the input of entire node set.

		The inference code is written in a fashion that it could handle any number of nodes and
		layers.
		"""
		# During inference with sampling, multi-layer blocks are very inefficient because
		# lots of computations in the first few layers are repeated.
		# Therefore, we compute the representation of all nodes layer by layer.  The nodes
		# on each layer are of course splitted in batches.
		# TODO: can we standardize this?
		for l, layer in enumerate(self.layers):
			y = torch.zeros(g.num_nodes(), self.n_hidden if l != len(self.layers) - 1 else self.n_classes)

			sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
			dataloader = dgl.dataloading.NodeDataLoader(
				g,
				torch.arange(g.num_nodes()),
				sampler,
				batch_size=args.batch_size,
				shuffle=True,
				drop_last=False,
				num_workers=args.num_workers)

			for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
				block = blocks[0]

				block = block.int().to(device)
				h = x[input_nodes].to(device)
				h = layer(block, h)
				if l != len(self.layers) - 1:
					h = self.activation(h)
					h = self.dropout(h)

				y[output_nodes] = h.cpu()

			x = y
		return y




def compute_acc(pred, labels):
	"""
	Compute the accuracy of prediction given the labels.
	"""
	labels = labels.long()
	# print('compute acc labels----------------------------------------------')
	# print(labels)
	# print('torch.argmax(pred, dim=1)')
	# print(torch.argmax(pred, dim=1))
	# print((torch.argmax(pred, dim=1) == labels))
	# print('compute acc labels------------------------------------------len(pred)')
	# print(len(pred))

	return (torch.argmax(pred, dim=1) == labels).data.float().sum() / len(pred)


def evaluate(data_loader, model_engine, val_g, val_nfeat, val_labels, val_nid, args, timers, device):
	"""
	Evaluate the model on the validation set specified by ``val_nid``.
	g : The entire graph.
	inputs : The features of all the nodes.
	labels : The labels of all the nodes.
	val_nid : the node Ids for validation.
	device : The GPU device to evaluate on.
	"""

	acc_res = 0

	# Turn on evaluation mode which disables dropout.
	model_engine.eval()
	with torch.no_grad():
		for step, (input_nodes, seeds, blocks) in enumerate(data_loader):
			# Load the input features as well as output labels
			batch_inputs, batch_labels = load_subtensor(val_nfeat, val_labels, seeds, input_nodes, device)
			blocks = [block.int().to(device) for block in blocks]
			# Compute loss and prediction
			batch_pred = model_engine(blocks, batch_inputs)
			# print(batch_pred.device)
			# print(type(batch_pred))
			# print(batch_pred.shape)
			# print(batch_labels.device)
			# print(type(batch_labels))
			# print(batch_labels.shape)
			# print('cur_acc = compute_acc(batch_pred[val_nid], batch_labels[val_nid])')
			cur_acc = compute_acc(batch_pred, batch_labels)
			# print(cur_acc)
			acc_res += cur_acc

	# Move model back to the train mode.
	model_engine.train()

	return acc_res/len(data_loader)


def load_subtensor(nfeat, labels, seeds, input_nodes, device):
	"""
	Extracts features and labels for a subset of nodes
	"""
	batch_inputs = nfeat[input_nodes].to(device)
	batch_labels = labels[seeds].to(device)
	return batch_inputs, batch_labels


#### Entry point
def run(args, device, data, tic):
	# Unpack data
	n_classes, train_g, val_g, test_g, train_nfeat, train_labels, val_nfeat, val_labels, test_nfeat, test_labels = data
	in_feats = train_nfeat.shape[1]
	train_nid = torch.nonzero(train_g.ndata['train_mask'], as_tuple=True)[0]
	val_nid = torch.nonzero(val_g.ndata['val_mask'], as_tuple=True)[0]
	test_nid = torch.nonzero(~(test_g.ndata['train_mask'] | test_g.ndata['val_mask']), as_tuple=True)[0]

	print("in_feats " + str(in_feats))
	# print("train_g.shape "+ str(train_g.shape))
	print("train_labels.shape " + str(train_labels.shape))
	# print("val_g.shape "+ str(val_g.shape))

	# Create PyTorch DataLoader for constructing blocks

	sampler = dgl.dataloading.MultiLayerNeighborSampler(
		[int(fanout) for fanout in args.fan_out.split(',')])
	training_dataloader = dgl.dataloading.NodeDataLoader(
		train_g,
		train_nid,
		sampler,
		batch_size=args.batch_size,
		shuffle=True,
		drop_last=False,
		num_workers=args.num_workers)
	evaluate_dataloader = dgl.dataloading.NodeDataLoader(
		val_g,
		val_nid,
		sampler,
		batch_size=args.batch_size,
		shuffle=True,
		drop_last=False,
		num_workers=args.num_workers)
	test_dataloader = dgl.dataloading.NodeDataLoader(
		test_g,
		test_nid,
		sampler,
		batch_size=args.batch_size,
		shuffle=True,
		drop_last=False,
		num_workers=args.num_workers)
	print("args.batch_size " + str(args.batch_size))

	# Define model and optimizer
	net = SAGE(in_feats, args.num_hidden, n_classes, args.num_layers, F.relu, args.dropout, args.aggre)

	print(net)
	parameters = filter(lambda p: p.requires_grad, net.parameters())
	model_engine, optimizer, _, __ = deepspeed.initialize(
		args=args, model=net, model_parameters=parameters)

	device = model_engine.local_rank

	# model = model.to(device)
	loss_fcn = nn.CrossEntropyLoss()
	optimizer = optim.Adam(net.parameters(), lr=args.lr)

	# Training loop
	# train_stream = torch.cuda.stream()
	# data_loading_stream = torch.cuda.stream()
	param_groups_list = []
	gradients = []*5

	avg = 0
	iter_tput = []
	avg_step_data_trans_time_list = []
	avg_step_GPU_train_time_list = []
	avg_step_time_list = []

	for epoch in range(args.num_epochs):
		# Loop over the dataloader to sample the computation dependency graph as a list of blocks
		running_loss = 0
		print("*"*30 + str(epoch)+" epoch ")
		tic_step = time.time()
		print(' the length of training loader ------------------------------------------------------------')
		print(len(training_dataloader))
		print(len(evaluate_dataloader))
		print(len(test_dataloader))
		tic = time.time()
		step_time_list = []
		step_data_trans_time_list = []
		step_GPU_train_time_list = []
		start = torch.cuda.Event(enable_timing=True)
		end = torch.cuda.Event(enable_timing=True)
		# start.record()
		tic_step = time.time()
		torch.cuda.synchronize()
		model_engine.zero_grad()
		for step, (input_nodes, seeds, blocks) in enumerate(training_dataloader): # dgl graphsage examples
			print(
				"\n   ***************************     step   " + str(step) + "   *************************************")
			# get_memory("-----------------------------------------after start a new step")
			torch.cuda.synchronize()
			start.record()
			see_memory_usage("-----------------------------------------step start------------------------")

			batch_inputs, batch_labels = load_subtensor(train_nfeat, train_labels, seeds, input_nodes, device)
			see_memory_usage("-----------------------------------------before blocks to device")
			blocks = [block.int().to(device) for block in blocks]
			see_memory_usage("-----------------------------------------after blocks to device")

			torch.cuda.synchronize()  # wait for move to complete
			end.record()

			torch.cuda.synchronize()
			step_data_trans_time_list.append(start.elapsed_time(end))

			start1 = torch.cuda.Event(enable_timing=True)
			end1 = torch.cuda.Event(enable_timing=True)
			start1.record()
			# Compute loss and prediction
			batch_pred = model_engine(blocks, batch_inputs)
			see_memory_usage("-----------------------------------------after batch train")
			loss = loss_fcn(batch_pred, batch_labels)
			see_memory_usage("-----------------------------------------after batch loss")

			model_engine.backward(loss)
			see_memory_usage("-----------------------------------------after batch loss backward")
			# model_engine.step()
			# see_memory_usage("-----------------------------------------after batch engine step")

			torch.cuda.synchronize()  # wait for all training steps to complete
			end1.record()
			torch.cuda.synchronize()

			step_GPU_train_time_list.append(start1.elapsed_time(end1))

			torch.cuda.synchronize()
			step_time = time.time() - tic_step
			step_time_list.append(step_time)

			iter_tput.append(len(seeds) / (time.time() - tic_step))
			if step % args.log_every == 0:
				acc = compute_acc(batch_pred, batch_labels)
				gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
				print(
					'Epoch {:05d} | Step {:05d} | Loss {:.4f} | Train Acc {:.4f} | Speed (samples/sec) {:.4f} | GPU {:.1f} MB'.format(
						epoch, step, loss.item(), acc.item(), np.mean(iter_tput[3:]), gpu_mem_alloc))

			tic_step = time.time()

		model_engine.step()
		see_memory_usage("-----------------------------------------after batch engine step")

		if len(step_data_trans_time_list[5:]) > 0:
			avg_iteration_time = sum(step_data_trans_time_list[5:]) / len(step_data_trans_time_list[5:])
			print('avg iteration(step) data from cpu to GPU time:%.8f ms' % (avg_iteration_time))
			avg_step_data_trans_time_list.append(avg_iteration_time)

			avg_iteration_gpu_time = sum(step_GPU_train_time_list[5:]) / len(step_GPU_train_time_list[5:])
			print('avg iteration GPU training time:%.8f ms' % (avg_iteration_gpu_time))
			avg_step_GPU_train_time_list.append(avg_iteration_gpu_time)

			avg_step_time = sum(step_time_list[5:]) / len(step_time_list[5:])
			print('avg iteration (step) total cpu time:%.8f ms' % (avg_step_time * 1000))
			avg_step_time_list.append(avg_step_time)


		toc = time.time()
		print('Epoch Time(s): {:.4f}'.format(toc - tic))
		if epoch >= 5:
			avg += toc - tic
		if epoch % args.eval_every == 0 and epoch != 0:
			eval_acc = evaluate(evaluate_dataloader, model_engine, val_g, val_nfeat, val_labels, val_nid, args, timers, device)
			print('Eval Acc {:.4f}'.format(eval_acc))
			test_acc = evaluate(test_dataloader, model_engine, test_g, test_nfeat, test_labels, test_nid, args, timers, device)
			print('Test Acc: {:.4f}'.format(test_acc))

	print('Avg epoch time(time.time()): {} ms'.format(avg * 1000 / (epoch - 4)))
	if len(avg_step_data_trans_time_list) > 0:
		total_avg_iteration_time = sum(avg_step_data_trans_time_list) / len(avg_step_data_trans_time_list)
		print('total avg iteration(step) data from cpu to GPU time:%.8f ms' % (total_avg_iteration_time))

		total_avg_iteration_gpu_time = sum(avg_step_GPU_train_time_list) / len(avg_step_GPU_train_time_list)
		print('total avg iteration GPU training time:%.8f ms' % (total_avg_iteration_gpu_time))

		total_avg_step_time_list = sum(avg_step_time_list) / len(avg_step_time_list)
		print('total avg iteration (step) total cpu time:%.8f ms' % (total_avg_step_time_list * 1000))



if __name__ == '__main__':
	# get_memory("-----------------------------------------main_start***************************")
	timers = Timers()
	tt = time.time()
	print("main start at this time " + str(tt))
	# argparser = argparse.ArgumentParser("multi-gpu training")
	argparser = argparse.ArgumentParser("single-gpu training")
	argparser.add_argument('--local_rank', type=int, default=0,
						   help="GPU device ID. Use -1 for CPU training")
	# argparser.add_argument('--gpu', type=int, default=0,
	# 					   help="GPU device ID. Use -1 for CPU training")
	argparser.add_argument('--dataset', type=str, default='ogbn-products')
	argparser.add_argument('--aggre', type=str, default='lstm')
	argparser.add_argument('--seed', type=int, default=1236)

	argparser.add_argument('--num-epochs', type=int, default=101)
	argparser.add_argument('--num-hidden', type=int, default=16)
	argparser.add_argument('--num-layers', type=int, default=2)
	argparser.add_argument('--fan-out', type=str, default='10,25')

	argparser.add_argument('--batch-size', type=int, default=196615)
	# argparser.add_argument('--batch-size', type=int, default=98308)
	# argparser.add_argument('--batch-size', type=int, default=49154)
	# argparser.add_argument('--batch-size', type=int, default=24577)
	# argparser.add_argument('--batch-size', type=int, default=12289)
	# argparser.add_argument('--batch-size', type=int, default=6145)
	# argparser.add_argument('--batch-size', type=int, default=3000)
	# argparser.add_argument('--batch-size', type=int, default=1500)

	argparser.add_argument('--log-every', type=int, default=20)
	argparser.add_argument('--eval-every', type=int, default=100)
	argparser.add_argument('--lr', type=float, default=0.003)
	argparser.add_argument('--dropout', type=float, default=0.5)
	argparser.add_argument('--num-workers', type=int, default=4,
						   help="Number of sampling processes. Use 0 for no extra process.")
	argparser.add_argument('--inductive', action='store_true',
						   help="Inductive learning setting")
	argparser.add_argument('--data-cpu', action='store_true',
						   help="By default the script puts all node features and labels "
								"on GPU when using it to save time for data copy. This may "
								"be undesired if they cannot fit in GPU memory at once. "
								"This flag disables that.")
	parser = deepspeed.add_config_arguments(argparser)
	args = parser.parse_args()

	if args.local_rank >= 0:
		device = torch.device('cuda:%d' % args.local_rank)
	else:
		device = torch.device('cpu')
	set_seed(args)
	# get_memory("-----------------------------------------before load_ogb***************************")
	t2 = ttt(tt, "before load_ogb")
	if args.dataset == 'reddit':
		g, n_classes = load_reddit()
	if args.dataset == 'ogbn-products':
		g, n_classes = load_ogb(args.dataset)
		print('#nodes:', g.number_of_nodes())
		print('#edges:', g.number_of_edges())
		print('#classes:', n_classes)
	# get_memory("-----------------------------------------after load_ogb***************************")

	# if args.dataset in ['arxiv', 'collab', 'citation', 'ddi', 'protein', 'ppa', 'reddit.dgl','products']:
	#     g, n_classes = load_data(args.dataset)
	else:
		raise Exception('unknown dataset')
	# see_memory_usage("-----------------------------------------after data to cpu------------------------")
	t3 = ttt(t2, "after load_ogb")
	if args.inductive:
		train_g, val_g, test_g = inductive_split(g)
		train_nfeat = train_g.ndata.pop('features')
		val_nfeat = val_g.ndata.pop('features')
		test_nfeat = test_g.ndata.pop('features')
		train_labels = train_g.ndata.pop('labels')
		val_labels = val_g.ndata.pop('labels')
		test_labels = test_g.ndata.pop('labels')

	else:
		train_g = val_g = test_g = g
		train_nfeat = val_nfeat = test_nfeat = g.ndata.pop('features')
		train_labels = val_labels = test_labels = g.ndata.pop('labels')
	# get_memory("-----------------------------------------after inductive else***************************")
	t4 = ttt(t3, "after inductive else")

	if not args.data_cpu:
		train_nfeat = train_nfeat.to(device)
		train_labels = train_labels.to(device)
	# get_memory("-----------------------------------------after label***************************")
	t5 = ttt(t4, "after label")
	# Create csr/coo/csc formats before launching training processes with multi-gpu.
	# This avoids creating certain formats in each sub-process, which saves momory and CPU.
	train_g.create_formats_()
	# get_memory("-----------------------------------------after  train_g.create_formats_()***************************")
	val_g.create_formats_()
	# get_memory("-----------------------------------------after  train_g.create_formats_()***************************")
	test_g.create_formats_()
	# get_memory("-----------------------------------------before pack data***************************")
	t6 = ttt(t5, "after train_g.create_formats_()")
	# see_memory_usage("-----------------------------------------after model to gpu------------------------")
	# Pack data
	data = n_classes, train_g, val_g, test_g, train_nfeat, train_labels, \
		   val_nfeat, val_labels, test_nfeat, test_labels
	# get_memory("-----------------------------------------after pack data***************************")
	# t7 = ttt(t6, "after pack data")
	run(args, device, data, t6)