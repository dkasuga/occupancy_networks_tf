import tensorflow as tf
import numpy as np

from im2mesh import config
from im2mesh.checkpoints import CheckpointIO
from im2mesh.utils.io import export_pointcloud
'''
TODO: pytorch codes -> tensorflow
'''


class PSGNPreprocessor:
    ''' Point Set Generation Networks (PSGN) preprocessor class.

    Args:
        cfg_path (str): path to config file
        pointcloud_n (int): number of output points
        dataset (dataset): dataset
        model_file (str): model file
    '''

    def __init__(self,
                 cfg_path,
                 pointcloud_n,
                 dataset=None,
                 model_file=None):
        self.cfg = config.load_config(cfg_path, 'configs/default.yaml')
        self.pointcloud_n = pointcloud_n
        self.dataset = dataset
        self.model = config.get_model(self.cfg, dataset)

        # Output directory of psgn model
        out_dir = self.cfg['training']['out_dir']
        # If model_file not specified, use the one from psgn model
        if model_file is None:
            model_file = self.cfg['test']['model_file']
        # Load model
        self.checkpoint_io = CheckpointIO(out_dir, model=self.model)
        self.checkpoint_io.load(model_file)

    def __call__(self, inputs):
        points = self.model(inputs, training=False)

        batch_size = points.shape[0]
        T = points.shape[1]

        # Subsample points if necessary
        if T != self.pointcloud_n:
            idx = np.random.randint(low=0,
                                    high=T,
                                    size=(batch_size, self.pointcloud_n))
            idx = tf.convert_to_tensor([:, :, None])
            idx = tf.broadcast_to(
                idx, shape=[batch_size, self.pointcloud_n, 3])
            points = tf.gather(points, indices=idx, batch_dims=1)

        return points
