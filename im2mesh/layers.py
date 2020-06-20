import tensorflow as tf
import tensorflow_addons as tfa
'''
TODO: reset_parameters()
How should I initialize weights of layers as custom weight?
'''


# Resnet Blocks
class ResnetBlockFC(tf.keras.Model):
    ''' Fully connected ResNet Block class.

    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    '''

    def __init__(self, size_in, size_out=None, size_h=None):
        super().__init__()
        # Attributes
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.fc_0 = tf.keras.layers.Dense(size_h)
        self.fc_1 = tf.keras.layers.Dense(size_out, kernel_initializer='zeros')

        self.actvn = tf.keras.layers.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = tf.keras.layers(size_out, use_bias=False)

    def call(self, x):
        net = self.fc_0(self.actvn(x))
        dx = self.fc_1(self.actvn(net))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx


class CResnetBlockConv1d(tf.keras.Model):
    ''' Conditional batch normalization-based Resnet block class.

    Args:
        c_dim (int): dimension of latend conditioned code c
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
        norm_method (str): normalization method
        legacy (bool): whether to use legacy blocks
    '''

    def __init__(self,
                 c_dim,
                 size_in,
                 size_h=None,
                 size_out=None,
                 norm_method='batch_norm',
                 legacy=False):
        super().__init__()
        # Attributes
        if size_h is None:
            size_h = size_in
        if size_out is None:
            size_out = size_in

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        if not legacy:
            self.bn_0 = CBatchNorm1d(c_dim, size_in, norm_method=norm_method)
            self.bn_1 = CBatchNorm1d(c_dim, size_h, norm_method=norm_method)
        else:
            self.bn_0 = CBatchNorm1d_legacy(c_dim,
                                            size_in,
                                            norm_method=norm_method)
            self.bn_1 = CBatchNorm1d_legacy(c_dim,
                                            size_h,
                                            norm_method=norm_method)

        self.fc_0 = tf.keras.layers.Conv1D(size_h, 1)
        self.fc_1 = tf.keras.layers.Conv1D(size_out,
                                           1,
                                           kernel_initializer='zeros')
        self.actvn = tf.keras.layers.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = tf.keras.layers.Conv1d(size_out, 1, use_bias=False)

    def call(self, x, c):
        net = self.fc_0(self.actvn(self.bn_0(x, c)))
        dx = self.fc_1(self.actvn(self.bn_1(net, c)))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx


class ResnetBlockConv1d(tf.keras.Model):
    ''' 1D-Convolutional ResNet block class.

    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    '''

    def __init__(self, size_in, size_h=None, size_out=None):
        super().__init__()
        # Attributes
        if size_h is None:
            size_h = size_in
        if size_out is None:
            size_out = size_in

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.bn_0 = tf.keras.layers.BatchNormalization()
        self.bn_1 = tf.keras.layers.BatchNormalization()

        self.fc_0 = tf.keras.layers.Conv1D(size_h, 1)
        self.fc_1 = tf.keras.layers.Conv1D(size_out,
                                           1,
                                           kernel_initializer='zeros')
        self.actvn = tf.keras.layers.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = tf.keras.layers.Conv1d(size_out, 1, use_bias=False)

    def call(self, x):
        net = self.fc_0(self.actvn(self.bn_0(x)))
        dx = self.fc_1(self.actvn(self.bn_1(net)))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx


# Utility modules
class AffineLayer(tf.keras.Model):
    ''' Affine layer class.

    Args:
        c_dim (tensor): dimension of latent conditioned code c
        dim (int): input dimension
    '''

    def __init__(self, c_dim, dim=3):
        super().__init__()
        self.c_dim = c_dim
        self.dim = dim
        # Submodules
        # how should I do to initialize bias as specific tensor value
        self.fc_A = tf.keras.layers.Dense(dim * dim,
                                          kernel_initializer='zeros')
        self.fc_b = tf.keras.layers.Dense(dim, kernel_initializer='zeros')
        # self.reset_parameters()

    # def reset_parameters(self):
    #     nn.init.zeros_(self.fc_A.weight)
    #     nn.init.zeros_(self.fc_b.weight)
    #     with torch.no_grad():
    #         self.fc_A.bias.copy_(torch.eye(3).view(-1))
    #         self.fc_b.bias.copy_(torch.tensor([0., 0., 2.]))

    def call(self, x, p):
        assert (x.shape[0] == p.shape[0])
        assert (p.shape[2] == self.dim)
        batch_size = x.size(0)
        A = tf.reshape(self.fc_A(x), [batch_size, 3, 3])
        b = tf.reshape(self.fc_b(x), [batch_size, 1, 3])
        out = tf.linalg.matmul(p, A) + b
        return out


class CBatchNorm1d(tf.keras.Model):
    ''' Conditional batch normalization layer class.

    Args:
        c_dim (int): dimension of latent conditioned code c
        f_dim (int): feature dimension
        norm_method (str): normalization method
    '''

    def __init__(self, c_dim, f_dim, norm_method='batch_norm'):
        super().__init__()
        self.c_dim = c_dim
        self.f_dim = f_dim
        self.norm_method = norm_method
        # Submodules
        self.conv_gamma = tf.keras.layers.Conv1D(f_dim, 1)
        self.conv_beta = tf.keras.layers.Conv1D(f_dim, 1)
        if norm_method == 'batch_norm':
            self.bn = tf.keras.layers.BatchNormalization(trainable=False)
        elif norm_method == 'instance_norm':
            '''
            tfa.layers.InstanceNormalization doesn't have 'trainable' arguments
            '''
            self.bn = tfa.layers.InstanceNormalization(
            )  # TODO trainable=False
        elif norm_method == 'group_norm':
            '''
            tfa.layers.GroupNormalization doesn't have 'trainable' arguments
            '''
            self.bn = tfa.layers.GroupNormalization()  # TODO trainable=False
        else:
            raise ValueError('Invalid normalization method!')
        # self.reset_parameters()

    # def reset_parameters(self):
    #     nn.init.zeros_(self.conv_gamma.weight)
    #     nn.init.zeros_(self.conv_beta.weight)
    #     nn.init.ones_(self.conv_gamma.bias)
    #     nn.init.zeros_(self.conv_beta.bias)

    def call(self, x, c):
        assert (x.shape[0] == c.shape[0])
        assert (c.shape[1] == self.c_dim)

        # c is assumed to be of size batch_size x c_dim x T
        if len(c.size()) == 2:
            c = tf.expand_dims(c, 2)

        # Affine mapping
        gamma = self.conv_gamma(c)
        beta = self.conv_beta(c)

        # Batchnorm
        net = self.bn(x)
        out = gamma * net + beta

        return out


class CBatchNorm1d_legacy(tf.keras.Model):
    ''' Conditional batch normalization legacy layer class.

    Args:
        c_dim (int): dimension of latent conditioned code c
        f_dim (int): feature dimension
        norm_method (str): normalization method
    '''

    def __init__(self, c_dim, f_dim, norm_method='batch_norm'):
        super().__init__()
        self.c_dim = c_dim
        self.f_dim = f_dim
        self.norm_method = norm_method
        # Submodules
        self.fc_gamma = tf.keras.layers.Dense(f_dim)
        self.fc_beta = tf.keras.layers.Dense(f_dim)
        if norm_method == 'batch_norm':
            self.bn = tf.keras.layers.BatchNormalization(trainable=False)
        elif norm_method == 'instance_norm':
            '''
            tfa.layers.InstanceNormalization doesn't have 'trainable' arguments
            '''
            self.bn = tfa.layers.InstanceNormalization(
            )  # TODO trainable=False
        elif norm_method == 'group_norm':
            '''
            tfa.layers.GroupNormalization doesn't have 'trainable' arguments
            '''
            self.bn = tfa.layers.GroupNormalization()  # TODO trainable=False
        else:
            raise ValueError('Invalid normalization method!')
        # self.reset_parameters()

    # def reset_parameters(self):
    #     nn.init.zeros_(self.fc_gamma.weight)
    #     nn.init.zeros_(self.fc_beta.weight)
    #     nn.init.ones_(self.fc_gamma.bias)
    #     nn.init.zeros_(self.fc_beta.bias)

    def call(self, x, c):
        batch_size = x.shape[0]
        # Affine mapping
        gamma = self.fc_gamma(c)
        beta = self.fc_beta(c)
        gamma = tf.reshape(gamma, [batch_size, self.f_dim, 1])
        beta = tf.reshape(beta, [batch_size, self.f_dim, 1])
        # Batchnorm
        net = self.bn(x)
        out = gamma * net + beta

        return out
