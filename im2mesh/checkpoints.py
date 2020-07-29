import os
import urllib

import tensorflow as tf
import numpy as np


class CheckpointIO(object):
    """ CheckpointIO class.

    It handles saving and loading checkpoints.

    Args:
        model (tf.keras.Model): model saved with the checkpoints
        optimizer (tf.keras.optimizers): optimizer saved with the checkpoints
        model_selection_sign (int): parameter needed for initializing metric_val_best
        checkpoint_dir (str): path where checkpoints are saved
    """

    def __init__(self, model, optimizer, model_selection_sign=1, checkpoint_dir="./chkpts"):
        self.ckpt = tf.train.Checkpoint(
            model=model, optimizer=optimizer, epoch_it=tf.Variable(-1, dtype=tf.int64), it=tf.Variable(-1, dtype=tf.int64), metric_val_best=tf.Variable(-model_selection_sign * np.inf, dtype=tf.float32))

        self.checkpoint_dir = checkpoint_dir
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

    def save(self, filename, epoch_it, it, loss_val_best, **kwargs):
        """ Saves the current module dictionary.

        Args:
            filename (str): name of output file
            epoch_it (tf.Variable): epoch saved
            it (tf.Variable): iteration saved
            loss_val_best(tf.Variable): metric_val_best saved
        """
        if not os.path.isabs(filename):
            filename = os.path.join(self.checkpoint_dir, filename)

        self.ckpt.epoch_it.assign(epoch_it)
        self.ckpt.it.assign(it)
        self.ckpt.metric_val_best.assign(loss_val_best)
        self.ckpt.save(filename)

    def load(self, filename=None):
        '''Loads a module dictionary from local file or url.

        Args:
            filename (str): name of saved module dictionary
        '''
        if filename is not None:
            if not os.path.isabs(filename):
                filename = os.path.join(self.checkpoint_dir, filename)
        else:
            filename = tf.train.latest_checkpoint(self.checkpoint_dir)

        if filename is not None:
            print(filename)
            print("=> Loading checkpoint from local file...")
            self.ckpt.restore(filename)
        else:
            raise FileExistsError

    # def load_url(self, url):
    #     '''Load a module dictionary from url.

    #     Args:
    #         url (str): url to saved model
    #     '''
    #     print(url)
    #     print('=> Loading checkpoint from url...')
    #     state_dict = model_zoo.load_url(url, progress=True)
    #     scalars = self.parse_state_dict(state_dict)
    #     return scalars

    # def parse_state_dict(self, state_dict):
    #     """Parse state_dict of model and return scalars.

    #     Args:
    #         state_dict (dict): State dict of model
    # """

    #     for k, v in self.module_dict.items():
    #         if k in state_dict:
    #             v.load_state_dict(state_dict[k])
    #         else:
    #             print("Warning: Could not find %s in checkpoint!" % k)
    #     scalars = {k: v for k, v in state_dict.items() if k not in self.module_dict}
    #     return scalars


# unnecessary?
# url pretrained model is only for pytorch model, not tensorflow
# def is_url(url):
#     scheme = urllib.parse.urlparse(url).scheme
#     return scheme in ('http', 'https')
