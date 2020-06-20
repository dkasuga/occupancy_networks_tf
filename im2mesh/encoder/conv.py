from im2mesh.common import normalize_imagenet
import tensorflow as tf


class ConvEncoder(tf.keras.Model):
    r''' Simple convolutional encoder network.

    It consists of 5 convolutional layers, each downsampling the input by a
    factor of 2, and a final fully-connected layer projecting the output to
    c_dim dimenions.

    Args:
        c_dim (int): output dimension of latent embedding
    '''

    def __init__(self, c_dim=128):
        super().__init__()
        self.conv0 = tf.keras.Conv2D(32, 3, stride=2)
        self.conv1 = tf.keras.Conv2D(64, 3, stride=2)
        self.conv2 = tf.keras.Conv2D(128, 3, stride=2)
        self.conv3 = tf.keras.Conv2D(256, 3, stride=2)
        self.conv4 = tf.keras.Conv2D(512, 3, stride=2)
        self.fc_out = tf.keras.Dense(c_dim)
        self.actvn = tf.keras.layers.ReLU()

    def call(self, x):
        batch_size = x.shape[0]

        net = self.conv0(x)
        net = self.conv1(self.actvn(net))
        net = self.conv2(self.actvn(net))
        net = self.conv3(self.actvn(net))
        net = self.conv4(self.actvn(net))
        net = net.view(batch_size, 512, -1).mean(2)
        net = tf.reshape(net, [batch_size, 512, -1])
        net = tf.math.reduce_mean(net, 2)

        out = self.fc_out(self.actvn(net))

        return out


# class Resnet18(nn.Module):
#     r''' ResNet-18 encoder network for image input.
#     Args:
#         c_dim (int): output dimension of the latent embedding
#         normalize (bool): whether the input images should be normalized
#         use_linear (bool): whether a final linear layer should be used
#     '''
#     def __init__(self, c_dim, normalize=True, use_linear=True):
#         super().__init__()
#         self.normalize = normalize
#         self.use_linear = use_linear
#         self.features = models.resnet18(pretrained=True)
#         self.features.fc = nn.Sequential()
#         if use_linear:
#             self.fc = nn.Linear(512, c_dim)
#         elif c_dim == 512:
#             self.fc = nn.Sequential()
#         else:
#             raise ValueError('c_dim must be 512 if use_linear is False')

#     def forward(self, x):
#         if self.normalize:
#             x = normalize_imagenet(x)
#         net = self.features(x)
#         out = self.fc(net)
#         return out

# class Resnet34(nn.Module):
#     r''' ResNet-34 encoder network.

#     Args:
#         c_dim (int): output dimension of the latent embedding
#         normalize (bool): whether the input images should be normalized
#         use_linear (bool): whether a final linear layer should be used
#     '''
#     def __init__(self, c_dim, normalize=True, use_linear=True):
#         super().__init__()
#         self.normalize = normalize
#         self.use_linear = use_linear
#         self.features = models.resnet34(pretrained=True)
#         self.features.fc = nn.Sequential()
#         if use_linear:
#             self.fc = nn.Linear(512, c_dim)
#         elif c_dim == 512:
#             self.fc = nn.Sequential()
#         else:
#             raise ValueError('c_dim must be 512 if use_linear is False')

#     def forward(self, x):
#         if self.normalize:
#             x = normalize_imagenet(x)
#         net = self.features(x)
#         out = self.fc(net)
#         return out


class Resnet50(tf.keras.Model):
    r''' ResNet-50 encoder network.

    Args:
        c_dim (int): output dimension of the latent embedding
        normalize (bool): whether the input images should be normalized
        use_linear (bool): whether a final linear layer should be used
    '''

    def __init__(self, c_dim, normalize=True, use_linear=True):
        super().__init__()
        self.normalize = normalize
        self.use_linear = use_linear
        self.features = models.resnet50(pretrained=True)
        self.features = tf.keras.applications.ResNet50(
            include_top=False)  # feature_extractor

        if use_linear:
            self.fc = tf.keras.layers(c_dim)
        elif c_dim == 2048:
            # self.fc = nn.Sequential() # original
            self.fc = tf.keras.Sequential()  # CHECK
        else:
            raise ValueError('c_dim must be 2048 if use_linear is False')

    def call(self, x):
        if self.normalize:
            x = normalize_imagenet(x)
        net = self.features(x)
        out = self.fc(net)
        return out


class Resnet101(tf.keras.Model):
    r''' ResNet-101 encoder network.

    Args:
        c_dim (int): output dimension of the latent embedding
        normalize (bool): whether the input images should be normalized
        use_linear (bool): whether a final linear layer should be used
    '''

    def __init__(self, c_dim, normalize=True, use_linear=True):
        super().__init__()
        self.normalize = normalize
        self.use_linear = use_linear
        self.features = models.resnet101(pretrained=True)
        self.features = tf.keras.applications.ResNet50(
            include_top=False)  # feature_extractor

        if use_linear:
            self.fc = tf.keras.layers(c_dim)
        elif c_dim == 2048:
            # self.fc = nn.Sequential() # original
            self.fc = tf.keras.Sequential()  # CHECK
        else:
            raise ValueError('c_dim must be 2048 if use_linear is False')

    def call(self, x):
        if self.normalize:
            x = normalize_imagenet(x)
        net = self.features(x)
        out = self.fc(net)
        return out
