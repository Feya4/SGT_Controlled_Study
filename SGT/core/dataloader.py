"""BRACS Dataset loader."""
import os
import h5py
import torch.utils.data
import numpy as np
from dgl.data.utils import load_graphs
from torch.utils.data import Dataset
from glob import glob 
import dgl 
from train_utils import parse_device

# [rerun-fix] removed hard-coded CUDA_VISIBLE_DEVICES='0,1,2,3'; choose the GPU from the shell instead.

DEVICE = parse_device()

_LABEL_MAP = None


def set_label_map(mapping):
    """Configure optional original-label to task-label mapping."""
    global _LABEL_MAP
    _LABEL_MAP = None if mapping is None else tuple(int(x) for x in mapping)


def _map_label(label):
    label = int(label)
    if _LABEL_MAP is None:
        return label
    if label < 0 or label >= len(_LABEL_MAP):
        raise ValueError("Label {} is outside the configured map".format(label))
    return _LABEL_MAP[label]




COLLATE_FN = {
    'DGLGraph': lambda x: dgl.batch(x),
    'Tensor': lambda x: x,
    'int': lambda x: torch.LongTensor(x).to(DEVICE)
}

def _dgl_has_devices():
    try:
        major, minor = (int(v) for v in dgl.__version__.split('.')[:2])
    except ValueError:
        return True
    return (major, minor) >= (0, 5)


def set_graph_on_cuda(graph):
    # [rerun-fix] DGL >= 0.5 refuses GPU features on a CPU graph. The model copies the tissue-graph features
    # back to the CPU before preprocessing, so the graph can stay on the CPU there; DGL 0.4.x is unchanged.
    if _dgl_has_devices():
        return graph
    cuda_graph = dgl.DGLGraph()
    cuda_graph.add_nodes(graph.number_of_nodes())
    cuda_graph.add_edges(graph.edges()[0], graph.edges()[1])
    for key_graph, val_graph in graph.ndata.items():
        tmp = graph.ndata[key_graph].clone()
        cuda_graph.ndata[key_graph] = tmp.to(DEVICE)
    for key_graph, val_graph in graph.edata.items():
        cuda_graph.edata[key_graph] = graph.edata[key_graph].clone().to(DEVICE)
    return cuda_graph

def h5_to_tensor(h5_path):
    h5_object = h5py.File(h5_path, 'r')
    out = torch.from_numpy(np.array(h5_object['assignment_matrix']))
    return out


class BRACSDataset(Dataset):
    """BRACS dataset."""

    def __init__(
            self,
            cg_path: str = None,
            tg_path: str = None,
            assign_mat_path: str = None,
            load_in_ram: bool = False,
    ):
        """
        BRACS dataset constructor.

        Args:
            cg_path (str, optional): Cell Graph path to a given split (eg, cell_graphs/test/). Defaults to None.
            tg_path (str, optional): Tissue Graph path. Defaults to None.
            assign_mat_path (str, optional): Assignment matrices path. Defaults to None.
            load_in_ram (bool, optional): Loading data in RAM. Defaults to False.
        """
        super(BRACSDataset, self).__init__()

        assert not (cg_path is None and tg_path is None), "You must provide path to at least 1 modality."

        self.cg_path = cg_path
        self.tg_path = tg_path
        self.assign_mat_path = assign_mat_path
        self.load_in_ram = load_in_ram

        if cg_path is not None:
            self._load_cg()

        if tg_path is not None:
            self._load_tg()

        if assign_mat_path is not None:
            self._load_assign_mat()


    def _load_cg(self):
        """
        Load cell graphs
        """
        self.cg_fnames = glob(os.path.join(self.cg_path, '*.bin'))

        self.cg_fnames.sort()
        self.num_cg = len(self.cg_fnames)
        if self.load_in_ram:
            cell_graphs = [load_graphs(fname) for fname in self.cg_fnames]  # [rerun-fix] fname is already a full path
            self.cell_graphs = [entry[0][0] for entry in cell_graphs]
            self.cell_graph_labels = [entry[1]['label'].item() for entry in cell_graphs]

    def _load_tg(self):
        """
        Load tissue graphs
        """
        self.tg_fnames = glob(os.path.join(self.tg_path, '*.bin'))

        self.tg_fnames.sort()
        self.num_tg = len(self.tg_fnames)
        if self.num_tg == 0:
            raise FileNotFoundError('No *.bin tissue graphs found in {}'.format(self.tg_path))
        if self.load_in_ram:
            tissue_graphs = [load_graphs(fname) for fname in self.tg_fnames]  # [rerun-fix] fname is already a full path
            self.tissue_graphs = [entry[0][0] for entry in tissue_graphs]
            self.tissue_graph_labels = [entry[1]['label'].item() for entry in tissue_graphs]
            # [rerun-fix] original never loaded the weights in RAM mode -> NameError on `w`.
            self.tissue_graph_weights = [self._load_weight(fname) for fname in self.tg_fnames]

    @staticmethod
    def _weight_path(tg_fname):
        """.../tissue_graphs/<split>/<name>.bin -> .../weights/<split>/<name>.pt"""
        head, split_dir = os.path.split(os.path.dirname(tg_fname))
        root, tg_dirname = os.path.split(head)
        name = os.path.basename(tg_fname).replace('.bin', '.pt')
        return os.path.join(root, 'weights', split_dir, name)

    def _load_weight(self, tg_fname):
        w_name = self._weight_path(tg_fname)
        if not os.path.isfile(w_name):
            raise FileNotFoundError(
                'Missing connectivity weight file {}. Run build_weights.py first '
                '(or place the official weights there).'.format(w_name))
        return torch.load(w_name, map_location='cpu')

    def _load_assign_mat(self):
        """
        Load assignment matrices 
        """
        self.assign_fnames = glob(os.path.join(self.assign_mat_path, '*.h5'))
        self.assign_fnames = self.remove_fail(self.assign_fnames)

        self.assign_fnames.sort()
        self.num_assign_mat = len(self.assign_fnames)
        if self.load_in_ram:
            self.assign_matrices = [
                h5_to_tensor(os.path.join(self.assign_mat_path, fname)).float().t()
                    for fname in self.assign_fnames
            ]

    def __getitem__(self, index):
        """
        Get an example.
        Args:
            index (int): index of the example.
        """

        
        if hasattr(self, 'num_tg') and hasattr(self, 'num_cg'):
            if self.load_in_ram:
                cg = self.cell_graphs[index]
                tg = self.tissue_graphs[index]
                assign_mat = self.assign_matrices[index]
                assert self.cell_graph_labels[index] == self.tissue_graph_labels[index], "The CG and TG are not the same. There was an issue while creating HACT."
                label = self.cell_graph_labels[index]
            else:
                cg, label = load_graphs(self.cg_fnames[index])
                cg = cg[0]
                label = label['label'].item()
                tg, _ = load_graphs(self.tg_fnames[index])
                tg = tg[0]
                assign_mat = h5_to_tensor(self.assign_fnames[index]).float().t()

            cg = set_graph_on_cuda(cg) if torch.cuda.is_available() else cg
            tg = set_graph_on_cuda(tg) if torch.cuda.is_available() else tg
            assign_mat = assign_mat.to(DEVICE) if torch.cuda.is_available() else assign_mat

            return cg, tg, assign_mat, _map_label(label)

        
        elif hasattr(self, 'num_tg'):
            if self.load_in_ram:
                tg = self.tissue_graphs[index]
                label = self.tissue_graph_labels[index]
                w = self.tissue_graph_weights[index]
            else:
                tg, label = load_graphs(self.tg_fnames[index])
                label = label['label'].item()
                tg = tg[0]
                w = self._load_weight(self.tg_fnames[index])
            tg = set_graph_on_cuda(tg) if torch.cuda.is_available() else tg
            return tg, w, _map_label(label)

        
        else:
            if self.load_in_ram:
                cg = self.cell_graphs[index]
                label = self.cell_graph_labels[index]
            else:
                cg, label = load_graphs(self.cg_fnames[index])
                label = label['label'].item()
                cg = cg[0]
            cg = set_graph_on_cuda(cg) if torch.cuda.is_available() else cg
            return cg, _map_label(label)

    def __len__(self):
        """Return the number of samples in the BRACS dataset."""
        if hasattr(self, 'num_cg'):
            return self.num_cg
        else:
            return self.num_tg


def collate(batch):
    """
    Collate a batch.
    Args:
        batch (torch.tensor): a batch of examples.
    Returns:
        data: (tuple)
        labels: (torch.LongTensor)
    """
    def collate_fn(batch, id, type):
        return COLLATE_FN[type]([example[id] for example in batch])

    
    num_modalities = len(batch[0])  
    batch = tuple([collate_fn(batch, mod_id, type(batch[0][mod_id]).__name__)
                  for mod_id in range(num_modalities)])

    return batch


def make_data_loader(
        batch_size,
        shuffle=True,
        num_workers=0,
        **kwargs
    ):
    """
    Create a BRACS data loader.
    """

    dataset = BRACSDataset(**kwargs)
    dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate
        )

    return dataloader