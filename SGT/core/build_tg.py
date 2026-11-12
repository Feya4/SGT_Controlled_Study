import os
from glob import glob
import argparse
from PIL import Image
import numpy as np
from tqdm import tqdm
import torch 
from dgl.data.utils import save_graphs
import h5py

from histocartography.preprocessing import (
    VahadaneStainNormalizer,         
    NucleiExtractor,                 
    DeepFeatureExtractor,            
    KNNGraphBuilder,                 
    ColorMergedSuperpixelExtractor,  
    DeepFeatureExtractor,            
    RAGGraphBuilder,                 
    AssignmnentMatrixBuilder         
)


TUMOR_TYPE_TO_LABEL = {
    'N': 0,
    'PB': 1,
    'UDH': 2,
    'ADH': 3,
    'FEA': 4,
    'DCIS': 5,
    'IC': 6
}

MIN_NR_PIXELS = 50000
MAX_NR_PIXELS = 50000000
# [rerun-fix] was '../data/target.png', which does not exist when run from core/.
STAIN_NORM_TARGET_IMAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'target.png')


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image_path',
        type=str,
        help='path to one BRACS split, e.g. .../previous_version/train',
        default='',
        required=False
    )
    parser.add_argument(
        '--save_path',
        type=str,
        help='path to save the TG graphs.',
        default='',
        required=False
    )
    # [rerun-fix] explicit split; the original guessed it from the path string.
    parser.add_argument(
        '--split',
        type=str,
        choices=['train', 'val', 'test'],
        default=None,
        help='split name; if omitted it is guessed from --image_path.'
    )
    return parser.parse_args()

class TgBuilding:

    def __init__(self):
        
        self.normalizer = VahadaneStainNormalizer(target_path=STAIN_NORM_TARGET_IMAGE)

        self._build_tg_builders()

        self.image_ids_failing = []
        
        self.knn_graph_builder = KNNGraphBuilder(k=5, thresh=50, add_loc_feats=True)

    def _build_tg_builders(self):
        
        self.tissue_detector = ColorMergedSuperpixelExtractor(
            superpixel_size=500,
            compactness=20,
            blur_kernel_size=1,
            threshold=0.05,
            downsampling_factor=4
        )
        
        self.tissue_feature_extractor = DeepFeatureExtractor(
            architecture='resnet34',
            patch_size=144,
            resize_size=224
        )

        self.rag_graph_builder = RAGGraphBuilder(add_loc_feats=True)

    def _build_tg(self, image):
        superpixels, _ = self.tissue_detector.process(image)
        features = self.tissue_feature_extractor.process(image, superpixels)
        graph = self.rag_graph_builder.process(superpixels, features)
        return graph, superpixels

    def process(self, image_path, save_path, split):
        
        subdirs = os.listdir(image_path)
        image_fnames = []
        for subdir in (subdirs + ['']):  
            image_fnames += glob(os.path.join(image_path, subdir, '*.png'))

        print('*** Start analysing {} images ***'.format(len(image_fnames)))
        for image_path in tqdm(image_fnames):

            
            _, image_name = os.path.split(image_path)
            image = np.array(Image.open(image_path))
            nr_pixels = image.shape[0] * image.shape[1]
            image_label = TUMOR_TYPE_TO_LABEL[image_name.split('_')[2]]
            tg_out = os.path.join(save_path, 'tissue_graphs', split, image_name.replace('.png', '.bin'))

            # [rerun-fix] original was `if not self._valid_image(nr_pixels):`, which
            # skipped every valid image and only processed too-small/too-large ones.
            # Restored the HACT-Net logic: skip existing outputs, keep valid sizes.
            if not self._exists(tg_out) and self._valid_image(nr_pixels):

                try: 
                    image = self.normalizer.process(image)
                except:
                    print('Warning: {} failed during stain normalization.'.format(image_path))
                    self.image_ids_failing.append(image_path)
                    pass


                try: 
                    tissue_graph, tissue_map = self._build_tg(image)
                    save_graphs(
                        filename=tg_out,
                        g_list=[tissue_graph],
                        labels={"label": torch.tensor([image_label])}
                    )
                except:
                    print('Warning: {} failed during tissue graph generation.'.format(image_path))
                    self.image_ids_failing.append(image_path)
                    pass
            else:
                print('Image:', image_path, ' was already processed or is too large/small.')

        print('Out of {} images, {} successful graph generations.'.format(
            len(image_fnames),
            len(image_fnames) - len(self.image_ids_failing)
        ))
        print('Failing IDs are:', self.image_ids_failing)

    def _valid_image(self, nr_pixels):
        if nr_pixels > MIN_NR_PIXELS and nr_pixels < MAX_NR_PIXELS:
            return True
        return False

    def _exists(self,tg_out):
        if os.path.isfile(tg_out):
            return True
        return False


if __name__ == "__main__":

    args = parse_arguments()
    if not os.path.isdir(args.image_path) or not os.listdir(args.image_path):
        raise ValueError("Data directory is either empty or does not exist.")

    split = args.split
    if split is None:
        last = os.path.basename(os.path.normpath(args.image_path)).lower()
        if 'train' in last:
            split = 'train'
        elif 'val' in last:
            split = 'val'
        else:
            split = 'test'
    print('*** Split: {} | images from: {} ***'.format(split, args.image_path))

    os.makedirs(os.path.join(args.save_path, 'tissue_graphs', split), exist_ok=True)

    tg_builder = TgBuilding()
    tg_builder.process(args.image_path, args.save_path, split)
