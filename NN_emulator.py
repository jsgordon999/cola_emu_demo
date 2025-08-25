import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, callbacks

# Custom activation layer 
class CustomActivationLayer(layers.Layer):
    def __init__(self, units, name=None):
        super().__init__(name=name)
        self.units = units
    def build(self, input_shape):
        self.beta = self.add_weight(name="beta", shape=(self.units,), initializer="random_normal", trainable=True)
        self.gamma = self.add_weight(name="gamma", shape=(self.units,), initializer="random_normal", trainable=True)
        super().build(input_shape)
    def call(self, x):
        return (self.gamma + tf.sigmoid(self.beta * x) * (1.0 - self.gamma)) * x
    def get_config(self):
        cfg = super().get_config()
        cfg.update({"units": self.units})
        return cfg

# Build Model
def build_mlp(input_dim, output_dim, hidden=512, depth=3, lr=1e-3):
    x_in = layers.Input(shape=(input_dim,))
    x = x_in
    for i in range(depth):
        x = layers.Dense(hidden, name=f"dense_{i+1}")(x)
        x = CustomActivationLayer(hidden, name=f"custom_{i+1}")(x)
    out = layers.Dense(output_dim, name="out")(x)
    model = keras.Model(x_in, out, name="NNEmulator")
    model.compile(optimizer=optimizers.Adam(learning_rate=lr), loss="mse")
    return model

# LR Decay
class StepDecay(callbacks.Callback):
    def __init__(self, decayevery, decayrate):
        super().__init__()
        self.decayevery = decayevery
        self.decayrate = decayrate
    def on_epoch_begin(self, epoch, logs=None):
        if epoch > 0 and self.decayevery > 0 and (epoch % self.decayevery == 0):
            old_lr = float(keras.backend.get_value(self.model.optimizer.lr))
            keras.backend.set_value(self.model.optimizer.lr, old_lr * self.decayrate)

# Emulator class
class NNEmulator:
    def __init__(self, N_pc=50, pca_components=None, pca_mean=None,
                 hidden=512, depth=3, lr=1e-3, decayevery=0, decayrate=0.5, seed=1337):
        assert pca_components is not None and pca_mean is not None, \
            "Provide pca_components (N_pc, k_len*z_len) and pca_mean (k_len*z_len)."
        self.N_pc = N_pc
        self.ext_components = np.asarray(pca_components)  # (N_pc, D)
        self.ext_mean = np.asarray(pca_mean)              # (D,)

        self.hidden = hidden
        self.depth = depth
        self.lr = lr
        self.decayevery = decayevery
        self.decayrate = decayrate

        tf.random.set_seed(seed)
        np.random.seed(seed)

        self.model = None
        self.history_ = None

    # Train model    
    def fit(self, X_train, Y_train, X_val=None, Y_val=None, epochs=3000, batch_size=16, verbose=1, patience=50):

        assert Y_train.shape[1] == self.N_pc, f"Y_train must have shape (N_train, {self.N_pc})"
        input_dim = X_train.shape[1]

        self.model = build_mlp(input_dim, self.N_pc, hidden=self.hidden, depth=self.depth, lr=self.lr)
        cbs = []
        if self.decayevery and self.decayevery > 0:
            cbs.append(StepDecay(self.decayevery, self.decayrate))

        # Option for early stopping
        if X_val is not None and Y_val is not None:
            cbs.append(callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True))

        # Save training and validation loss history
        hist = self.model.fit(
            X_train, Y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=verbose,
            validation_data=(X_val, Y_val) if X_val is not None and Y_val is not None else None,
            callbacks=cbs
        )
        self.history_ = hist.history

        return self

    # Predict Q(k,z) = log(P_NL/P_L)
    def predict_Q(self, X, out_shape=None):
        Y_hat = self.model.predict(X, verbose=0)          
        Q = Y_hat @ self.ext_components + self.ext_mean  
        if out_shape is not None:
            Q = Q.reshape(out_shape)
        return Q

    # Save model
    def save(self, filepath):
        self.model.save(filepath)

    # Load model
    def load(self, filepath):
        # Extract units
        def _custom_activation_layer_from_config(**cfg):
            units = cfg.pop("units")  # required
            return CustomActivationLayer(units)

        self.model = keras.models.load_model(filepath, custom_objects={"CustomActivationLayer": _custom_activation_layer_from_config})
