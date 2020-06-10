import tensorflow as tf
import tensorflow_probability as tfp

from im2mesh.onet.models import encoder_latent, decoder


# Encoder latent dictionary
encoder_latent_dict = {
    "simple": encoder_latent.Encoder,
}

# Decoder dictionary
decoder_dict = {
    "simple": decoder.Decoder,
    "cbatchnorm": decoder.DecoderCBatchNorm,
    "cbatchnorm2": decoder.DecoderCBatchNorm2,
    "batchnorm": decoder.DecoderBatchNorm,
    "cbatchnorm_noresnet": decoder.DecoderCBatchNormNoResnet,
}


class OccupancyNetwork(tf.keras.Model):
    """ Occupancy Network class.

    Args:
        decoder (nn.Module): decoder network
        encoder (nn.Module): encoder network
        encoder_latent (nn.Module): latent encoder network
        p0_z (dist): prior distribution for latent code z
        device (device): torch device
    """

    def __init__(self, decoder, encoder=None, encoder_latent=None, p0_z=None):
        super().__init__()
        if p0_z is None:
            p0_z = tfp.distributions.Normal(loc=[], scale=[])

        self.decoder = decoder

        if encoder_latent is not None:
            self.encoder_latent = encoder_latent
        else:
            self.encoder_latent = None

        if encoder is not None:
            self.encoder = encoder
        else:
            self.encoder = None

        self.p0_z = p0_z

    def call(self, p, inputs, sample=True, **kwargs):
        """ Performs a forward pass through the network.

        Args:
            p (tensor): sampled points
            inputs (tensor): conditioning input
            sample (bool): whether to sample for z
        """
        batch_size = p.shape[0]
        c = self.encode_inputs(inputs)
        z = self.get_z_from_prior([batch_size], sample=sample)
        p_r = self.decode(p, z, c, **kwargs)
        return p_r

    def compute_elbo(self, p, occ, inputs, **kwargs):
        """ Computes the expectation lower bound.

        Args:
            p (tensor): sampled points
            occ (tensor): occupancy values for p
            inputs (tensor): conditioning input
        """
        c = self.encode_inputs(inputs)
        q_z = self.infer_z(p, occ, c, **kwargs)
        # reparameterize
        mean = q_z.mean()
        logvar = tf.math.log(q_z.variance())
        eps = tf.random.normal(shape=mean.shape)
        z = eps * tf.exp(logvar * 0.5) + mean

        p_r = self.decode(p, z, c, **kwargs)

        # todo
        # rec_error = -p_r.log_prob(occ).sum(dim=-1)  # todo log_prob
        rec_error = -tf.reduce_sum(p_r.log_prob(occ), axis=-1)  # todo log_prob
        # kl = dist.kl_divergence(q_z, self.p0_z).sum(dim=-1)
        kl = tf.reduce_sum(tfp.distributions.kl_divergence(q_z, self.p0_z), axis=-1)
        elbo = -rec_error - kl

        return elbo, rec_error, kl

    def encode_inputs(self, inputs):
        """ Encodes the input.

        Args:
            input (tensor): the input
        """

        if self.encoder is not None:
            c = self.encoder(inputs)
        else:
            # Return inputs?
            c = tf.zeros([inputs.shape[0], 0], tf.float32)

        return c

    def decode(self, p, z, c, **kwargs):
        """ Returns occupancy probabilities for the sampled points.

        Args:
            p (tensor): points
            z (tensor): latent code z
            c (tensor): latent conditioned code c
        """

        logits = self.decoder(p, z, c, **kwargs)
        p_r = tfp.distributions.Bernoulli(logits=logits)
        return p_r

    def infer_z(self, p, occ, c, **kwargs):
        """ Infers z.

        Args:
            p (tensor): points tensor
            occ (tensor): occupancy values for occ
            c (tensor): latent conditioned code c
        """
        if self.encoder_latent is not None:
            mean_z, logstd_z = self.encoder_latent(p, occ, c, **kwargs)
        else:
            batch_size = p.shape[0]
            mean_z = tf.zeros([batch_size, 0], tf.float32)
            logstd_z = tf.zeros([batch_size, 0], tf.float32)

        q_z = tfp.distributions.Normal(mean_z, tf.math.exp(logstd_z))
        return q_z

    def get_z_from_prior(self, shape=[], sample=True):
        """ Returns z from prior distribution.

        Args:
            size (Size): size of z
            sample (bool): whether to sample
        """
        if sample:
            z = self.p0_z.sample([])
        else:
            z = self.p0_z.mean()
            z = tf.broadcast_to(z, [*shape, *(z.shape.as_list())])

        return z