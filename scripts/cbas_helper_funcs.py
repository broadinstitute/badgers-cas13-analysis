# This script contains all the relevant functions (from the original cbas repository) needed to train a VAE and propose sequences 
# Citation for code: https://github.com/dhbrookes/CbAS/tree/master

# Due to tensorflow incompatibilities, the VAE couldn't be run directly within the same process,
# so this script is called by the main process to train the VAE and propose sequences.
 
import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from tensorflow import keras
from tensorflow.python.keras import backend as K
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Layer, Input, Lambda, Add, Multiply, Dense, Flatten, Concatenate, Reshape, Conv1D
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.python.keras import backend as K
import argparse
import pandas as pd
import numpy as np
import os
 

### Below code is from CbAS vae.py script:
"""
Module for extendable variational autoencoders. 

Some code adapted from Louis Tiao's blog: http://louistiao.me/
"""
class KLDivergenceLayer(Layer):
    """ Add KL divergence in latent layer to loss """

    
    def __init__(self, *args, **kwargs):
        self.is_placeholder = True
        super(KLDivergenceLayer, self).__init__(*args, **kwargs)

     
    def call(self, inputs, scale=1.):
        """ Add KL loss, then return inputs """

        mu, log_var = inputs
        inner = 1 + log_var - K.square(mu) - K.exp(log_var)

        # sum over dimensions of latent space
        kl_batch = -0.5 * K.sum(inner, axis=1)

        # add mean KL loss over batch
        self.add_loss(scale * K.mean(kl_batch, axis=0), inputs=inputs)
        return mu, log_var


class KLScaleUpdate(Callback):
    """ Callback for updating the scale of the the KL divergence loss

    See Bowman et. al (2016) for motivation on adjusting the scale of the
    KL loss. This class implements a sigmoidal growth, as in Bowman, et. al.

    """

    
    def __init__(self, scale, growth=0.01, start=0.001, verbose=True):
        super(KLScaleUpdate, self).__init__()
        self.scale_ = scale
        self.start_ = start
        self.growth_ = growth
        self.step_ = 0
        self.verbose_ = verbose

     
    def on_batch_begin(self, batch, logs=None):
        K.set_value(self.scale_, self._get_next_val(self.step_))
        self.step_ += 1

     
    def _get_next_val(self, step):
        return 1 - (1 / (1 + self.start_ * np.exp(step * self.growth_)))

     
    def on_epoch_begin(self, epoch, logs=None):
        if self.verbose_ > 0:
            print("KL Divergence weight: %.3f" % K.get_value(self.scale_))


class BaseVAE(object):
    """ Base class for Variational Autoencoders implemented in Keras

    The class is designed to connect user-specified encoder and decoder
    models via a Model representing the latent space

    """

    
    def __init__(self, input_shape, latent_dim, *args, **kwargs):
        self.latentDim_ = latent_dim
        self.inputShape_ = input_shape

        self.encoder_ = None
        self.decoder_ = None

        self.vae_ = None

     
    def build_encoder(self, *args, **kwargs):
        """ Build the encoder network as a keras Model

        The encoder Model must ouput the mean and log variance of
        the latent space embeddings. I.e. this model must output
        mu and Sigma of the latent space distribution:

                    q(z|x) = N(z| mu(x), Sigma(x))

        Sets the value of self.encoder_ to a keras Model

        """

        raise NotImplementedError

     
    def build_decoder(self, *args, **kwargs):
        """ Build the decoder network as a keras Model

        The input to the decoder must have the same shape as the latent
        space and the output must have the same shape as the input to
        the encoder.

        Sets the value of self.decoder_ to a keras Model

        """

        raise NotImplementedError

     
    def _build_latent_vars(self, mu_z, log_var_z, epsilon_std=1., kl_scale=1.):
        """ Build keras variables representing the latent space

        First, calculate the KL divergence from the input mean and log variance
        and add this to the model loss via a KLDivergenceLayer. Then sample an epsilon
        and perform a location-scale transformation to obtain the latent embedding, z.

        Args:
            epsilon_std: standard deviation of p(epsilon)
            kl_scale: weight of KL divergence loss

        Returns:
            Variables representing z and epsilon

        """
         
 
        # mu_z, log_var_z, kl_batch  = KLDivergenceLayer()([mu_z, log_var_z], scale=kl_scale)
        lmda_func = lambda inputs: -0.5 * K.sum(1 + inputs[1] - K.square(inputs[0]) - K.exp(inputs[1]), axis=1)

        kl_batch = Lambda(lmda_func, name='kl_calc')([mu_z, log_var_z])
        kl_batch = Reshape((1,), name='kl_reshape')(kl_batch)

        # get standard deviation from log variance:
        sigma_z = Lambda(lambda lv: K.exp(0.5 * lv))(log_var_z)

        # re-parametrization trick ( z = mu_z + eps * sigma_z)
        eps = Input(tensor=K.random_normal(stddev=epsilon_std,
                                           shape=(K.shape(mu_z)[0], self.latentDim_)))

        eps_z = Multiply()([sigma_z, eps])  # scale by epsilon sample
        z = Add()([mu_z, eps_z])

        return z, eps, kl_batch

     
    def _get_decoder_input(self, z, enc_in):
         
        return z

     
    def build_vae(self, epsilon_std=1., kl_scale=1.):
        """ Build the VAE

        Sets the value of self.vae_ to a keras model

        Args:
            epsilon_std (float): standard deviation of distribution used to sample z via the reparameratization trick
            kl_scale (float or keras.backend.Variable): weight of KL divergence loss

        """
         
        if self.encoder_ is None:
            raise TypeError("Encoder must be built before calling build_vae")
        if self.decoder_ is None:
            raise TypeError("Decoder must be built before calling build_vae")

        enc_in = self.encoder_.inputs
        mu_z, log_var_z = self.encoder_.outputs
        z, eps, kl_batch = self._build_latent_vars(mu_z, log_var_z, epsilon_std=epsilon_std, kl_scale=kl_scale)
        dec_in = self._get_decoder_input(z, enc_in)
        x_pred = self.decoder_(dec_in)
        self.vae_ = Model(inputs=enc_in + [eps], outputs=[x_pred, kl_batch], name='vae_base')
        #self.vae_ = Model(inputs=(enc_in,eps), outputs=[x_pred, kl_batch], name='vae_base')
        #self.vae_ = Model(inputs=(enc_in,eps), outputs=x_pred, name='vae_base')    


        # Is there a way to name these outputs?

     
    def plot_model(self, *args, **kwargs):
         
        keras.utils.plot_model(self.vae_, *args, **kwargs)
    
     
    def compile(self, *args, **kwargs):
        
        self.vae_.compile(*args, **kwargs)

     
    def fit(self, *args, **kwargs):  

        self.vae_.fit(*args, **kwargs)

     
    def save_all_weights(self, prefix):
         
        encoder_file = prefix + "_encoder.h5"
        decoder_file = prefix + "_decoder.h5"
        vae_file = prefix + "_vae.h5"

        self.encoder_.save_weights(encoder_file)
        self.decoder_.save_weights(decoder_file)
        self.vae_.save_weights(vae_file)

     
    def load_all_weights(self, prefix):
         
        encoder_file = prefix + "_encoder.h5"
        decoder_file = prefix + "_decoder.h5"
        vae_file = prefix + "_vae.h5"

        self.encoder_.load_weights(encoder_file)
        self.decoder_.load_weights(decoder_file)
        self.vae_.load_weights(vae_file)


class SimpleVAE(BaseVAE):
    """ Basic VAE where the encoder and decoder can be constructed from lists of layers """

    
    def __init__(self, input_shape, latent_dim, flatten=True, *args, **kwargs):
         
        super(SimpleVAE, self).__init__(input_shape=input_shape,
                                        latent_dim=latent_dim,
                                        *args, **kwargs)
        self.flatten_ = flatten
        self.encoderLayers_ = []
        self.decoderLayers_ = []

     
    def add_encoder_layer(self, layer):
        """ Append a keras Layer to self.encoderLayers_"""
         
        self.encoderLayers_.append(layer)

     
    def add_decoder_layer(self, layer):
        """ Append a keras Layer to self.decoderLayers_ """
         
        self.decoderLayers_.append(layer)

     
    def _build_encoder_inputs(self):
        """ BUILD (as opposed to get) the encoder inputs """
         
        x = Input(shape=self.inputShape_, name = 'input1') 
        return [x]

     
    def _build_decoder_inputs(self):
         
        z = Input(shape=(self.latentDim_,)) 
        return z

     
    def _edit_encoder_inputs(self, enc_in):
         
        if self.flatten_:
            h = Flatten()(enc_in[0])
        else:
            h = enc_in[0]
        return h

     
    def _edit_decoder_inputs(self, dec_in):
         
        return dec_in

     
    def build_encoder(self):
        """ Construct the encoder from list of layers

        After the final layer in self.encoderLayers_, two Dense layers
        are applied to output mu_z and log_var_z

        """
         
        if len(self.encoderLayers_) == 0:
            raise ValueError("Must add at least one encoder hidden layer")

        enc_in = self._build_encoder_inputs()
        h = self._edit_encoder_inputs(enc_in)
        for hid in self.encoderLayers_:
            h = hid(h)

        mu_z = Dense(self.latentDim_, name='mu_z')(h)
        log_var_z = Dense(self.latentDim_, name='log_var_z')(h)

        self.encoder_ = Model(inputs=enc_in, outputs=[mu_z, log_var_z], name='encoder')

     
    def build_decoder(self, decode_activation):
        """ Construct the decoder from list of layers

        After the final layer in self.decoderLayers_, a Dense layer is
        applied to output the final reconstruction

        Args:
            decode_activation: activation of the final decoding layer

        """
         
        if len(self.decoderLayers_) == 0:
            raise ValueError("Must add at least one decoder hidden layer")

        dec_in = self._build_decoder_inputs()
        h = self._edit_decoder_inputs(dec_in)
        for hid in self.decoderLayers_: 
            h = hid(h)

        x_pred = h
        self.decoder_ = Model(inputs=dec_in, outputs=x_pred, name='decoder')

### Code from cbas losses.py script
def identity_loss(y_true, y_pred):
    """Returns the predictions"""
    return y_pred

 
def summed_categorical_crossentropy(y_true, y_pred):
    """ Negative log likelihood of categorical distribution """
    return K.sum(K.categorical_crossentropy(y_true, y_pred), axis=-1)


### Code from Cbas util.py script
def build_vae(latent_dim, n_tokens=4, seq_length=33, enc1_units=50, eps_std=1., ):
    """Returns a compiled VAE model"""
    model = SimpleVAE(input_shape=(seq_length, n_tokens,),
                      latent_dim=latent_dim)

    
    # 

    # set encoder layers:
    model.encoderLayers_ = [
        Dense(units=enc1_units, activation='elu', name='e2'),
    ]

    # set decoder layers:
    model.decoderLayers_ = [
        Dense(units=enc1_units, activation='elu', name='d1'),
        Dense(units=n_tokens * seq_length, name='d3'),
        Reshape((seq_length, n_tokens), name='d4'),
        Dense(units=n_tokens, activation='softmax', name='d5'),
    ]

    # build models:
    kl_scale = K.variable(1.)
    model.build_encoder()
    model.build_decoder(decode_activation='softmax')
    model.build_vae(epsilon_std=eps_std, kl_scale=kl_scale)
  
    losses = [summed_categorical_crossentropy, identity_loss]
    #losses = [summed_categorical_crossentropy]  

    model.compile(optimizer='adam',
                  loss=losses)

    return model

def get_samples(Xt_p):
    """Samples from a categorical probability distribution specifying the probability
    of amino acids at each position in a sequence"""
    Xt_sampled = np.zeros_like(Xt_p)
    # Xt_p  = Xt_p[~np.isnan(Xt_p)]
    for i in range(Xt_p.shape[0]):
        for j in range(Xt_p.shape[1]):
            p = Xt_p[i, j]
            k = np.random.choice(range(len(p)), p=p)
            Xt_sampled[i, j, k] = 1.
    return Xt_sampled
 
def get_balaji_predictions(preds, Xt):
    """Given a set of predictors built according to the methods in 
    the Balaji Lakshminarayanan paper 'Simple and scalable predictive 
    uncertainty estimation using deep ensembles' (2017), returns the mean and
    variance of the total prediction."""
    M = len(preds)
    N = Xt.shape[0]
    means = np.zeros((M, N))
    variances = np.zeros((M, N))
    for m in range(M):
        y_pred = preds[m].predict(Xt)
        means[m, :] = y_pred[:, 0]
        variances[m, :] = np.log(1+np.exp(y_pred[:, 1])) + 1e-6
    mu_star = np.mean(means, axis=0)
    var_star = (1/M) * (np.sum(variances, axis=0) + np.sum(means**2, axis=0)) - mu_star**2
    return mu_star, var_star 




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode',
        help=("number of the seq to run"),
        type = int) 
    parser.add_argument('filepath',
        help=("in/out files"),
        type = str) 

    args = parser.parse_args()

    # The below code enables this script to interface with the cbas.py script:
    if args.mode == 1: 
        # Train VAE_0 and save weights
        if not os.path.exists(args.filepath):
            os.makedirs(args.filepath)

        X_train = np.load(f"{args.filepath}/X_train.npy")
        hyperparams = pd.read_pickle(f"{args.filepath}/hyperparams.pkl")
        vae_0 = build_vae(latent_dim=hyperparams.latent_dim,
                  n_tokens=4, 
                  seq_length=X_train.shape[1],
                  enc1_units=50)
        vae_0.fit(X_train, [X_train, np.zeros(X_train.shape[0])],
                        epochs=100, # changed from 100
                        batch_size=10,
                        verbose=0)
        vae_0.save_all_weights(f"{args.filepath}/vae0")
        vae_0.save_all_weights(f"{args.filepath}/prev_vae")

    elif args.mode == 2:
        # Propose sequences
        hyperparams = pd.read_pickle(f"{args.filepath}/hyperparams.pkl")

        seq_len_vae = np.load(f"{args.filepath}/X_train.npy").shape[1]
        vae = build_vae(latent_dim=hyperparams.latent_dim,
                  n_tokens=4, 
                  seq_length=seq_len_vae,
                  enc1_units=50)
        
        vae.load_all_weights(f"{args.filepath}/prev_vae")

        zt = np.random.randn(hyperparams.cycle_batch_size, hyperparams.latent_dim) 
        Xt_p = vae.decoder_.predict(zt)
        
        mask = np.isnan(Xt_p) | np.isinf(Xt_p)
        mask_any_nan_inf = np.any(mask, axis=(1, 2))
        Xt_p = Xt_p[~mask_any_nan_inf] 
        Xt = get_samples(Xt_p)
 
        np.save(f"{args.filepath}/proposals_onehot.npy", Xt)
        np.save(f"{args.filepath}/Xt_p.npy", Xt_p)

        # Getting the probs from the initial VAE
        vae_0 = build_vae(latent_dim=hyperparams.latent_dim,
                  n_tokens=4, 
                  seq_length=seq_len_vae,
                  enc1_units=50)
        vae_0.load_all_weights(f"{args.filepath}/vae0") 

        X0_p = vae_0.decoder_.predict(zt)

        np.save(f"{args.filepath}/X0_p.npy", X0_p)

    elif args.mode == 3:
        # Train VAE
        hyperparams = pd.read_pickle(f"{args.filepath}/hyperparams.pkl")
        X_cont_train = np.load(f"{args.filepath}/X_cont_train.npy")
        weights = np.load(f"{args.filepath}/weights.npy")
        seq_len_vae = np.load(f"{args.filepath}/X_train.npy").shape[1]

        vae = build_vae(latent_dim=hyperparams.latent_dim,
                  n_tokens=4, 
                  seq_length=seq_len_vae,
                  enc1_units=50)
        
        vae.load_all_weights(f"{args.filepath}/prev_vae")

        print(len(X_cont_train))
        vae.fit([X_cont_train], [X_cont_train, np.zeros(X_cont_train.shape[0])],
                  epochs=hyperparams.per_train_epochs,
                  batch_size=10,
                  shuffle=False,
                  sample_weight=[weights, weights],
                  verbose=0)  
        
        vae.save_all_weights(f"{args.filepath}/prev_vae")

    
