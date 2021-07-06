import dgl
import torch as th
from cpu_mem_usage import get_memory
import time
from ogb.nodeproppred import DglNodePropPredDataset
def ttt(tic, str1):
    toc = time.time()
    print(str1 + ' step Time(s): {:.4f}'.format(toc - tic))
    return toc

def load_reddit():
    from dgl.data import RedditDataset

    # load reddit data
    data = RedditDataset(self_loop=True)
    g = data[0]
    g.ndata['features'] = g.ndata['feat']
    g.ndata['labels'] = g.ndata['label']
    return g, data.num_labels

def load_ogb(name):

    tic_step = time.time()
    get_memory("-" * 40 + "---------------------from ogb.nodeproppred import DglNodePropPredDataset***************************")
    print('load', name)
    data = DglNodePropPredDataset(name=name)
    t1 = ttt(tic_step, "-"*40+"---------------------data = DglNodePropPredDataset(name=name)***************************")
    # get_memory("-"*40+"---------------------data = DglNodePropPredDataset(name=name)***************************")
    print('finish loading', name)
    splitted_idx = data.get_idx_split()
    t2 = ttt(t1,"-" * 40 + "---------------------splitted_idx = data.get_idx_split()***************************")
    # get_memory("-" * 40 + "---------------------splitted_idx = data.get_idx_split()***************************")
    graph, labels = data[0]
    # get_memory("-" * 40 + "---------------------graph, labels = data[0]***************************")
    t3 = ttt(t2, "-" * 40 + "---------------------graph, labels = data[0]***************************")
    print(labels)
    print(data[0])
    print(graph)
    labels = labels[:, 0]
    # get_memory("-" * 40 + "---------------------labels = labels[:, 0]***************************")
    t4 = ttt(t3, "-" * 40 + "---------------------labels = labels[:, 0]***************************")

    graph.ndata['features'] = graph.ndata['feat']
    # get_memory("-" * 40 + "---------------------graph.ndata['features'] = graph.ndata['feat']***************************")
    t5 = ttt(t4, "-" * 40 + "---------------------graph.ndata['features'] = graph.ndata['feat']***************************")
    graph.ndata['labels'] = labels
    t6 = ttt(t5, "-" * 40 + "---------graph.ndata['labels'] = labels******************")
    in_feats = graph.ndata['features'].shape[1]
    num_labels = len(th.unique(labels[th.logical_not(th.isnan(labels))]))

    # Find the node IDs in the training, validation, and test set.
    train_nid, val_nid, test_nid = splitted_idx['train'], splitted_idx['valid'], splitted_idx['test']
    t7 = ttt(t6, "-" * 40 + "---------train_nid, val_nid, test_nid = splitted_idx******************")
    # get_memory(
	    # "-" * 40 + "---------------------train_nid, val_nid, test_nid = splitted_idx***************************")
    train_mask = th.zeros((graph.number_of_nodes(),), dtype=th.bool)
    train_mask[train_nid] = True
    val_mask = th.zeros((graph.number_of_nodes(),), dtype=th.bool)
    val_mask[val_nid] = True
    test_mask = th.zeros((graph.number_of_nodes(),), dtype=th.bool)
    test_mask[test_nid] = True
    graph.ndata['train_mask'] = train_mask
    graph.ndata['val_mask'] = val_mask
    graph.ndata['test_mask'] = test_mask
    t8 = ttt(t7, "-" * 40 + "---------end of load ogb******************")
    # get_memory(
	    # "-" * 40 + "---------------------end of load ogb***************************")

    print('finish constructing', name)
    print('load ogb-products time total: '+ str(time.time()-tic_step))
    return graph, num_labels

def inductive_split(g):
    """Split the graph into training graph, validation graph, and test graph by training
    and validation masks.  Suitable for inductive models."""
    train_g = g.subgraph(g.ndata['train_mask'])
    val_g = g.subgraph(g.ndata['train_mask'] | g.ndata['val_mask'])
    test_g = g
    return train_g, val_g, test_g