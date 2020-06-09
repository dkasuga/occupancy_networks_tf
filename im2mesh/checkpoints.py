import os
import urllib

import tensorflow as tf


class CheckpointIO(object):
    """ CheckpointIO class.

    It handles saving and loading checkpoints.

    Args:
        checkpoint_dir (str): path where checkpoints are saved
    """

    def __init__(self, model, checkpoint_dir="./chkpts"):
        self.model = model  # include layers weights and optimizer
        self.checkpoint_dir = checkpoint_dir
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

    # def register_modules(self, **kwargs):
    #     """ Registers modules in current module dictionary.
    #     """
    #     self.module_dict.update(kwargs)

    def save(self, filename, **kwargs):
        """ Saves the current module dictionary.

        Args:
            filename (str): name of output file
        """
        if not os.path.isabs(filename):
            filename = os.path.join(self.checkpoint_dir, filename)

        tf.keras.models.save_model(self.model, filename)

    def load(self, filename):
        if not os.path.isabs(filename):
            filename = os.path.join(self.checkpoint_dir, filename)

        if os.path.exists(filename):
            print(filename)
            print("=> Loading checkpoint from local file...")
            new_model = tf.keras.models.load_model(filename)
            return new_model
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
