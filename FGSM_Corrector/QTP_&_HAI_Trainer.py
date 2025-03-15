# Step 1: Install Libraries
!pip install tensorflow==2.15.0 numpy tensorly scikit-learn

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
import tensorly as tl
from sklearn.preprocessing import StandardScaler
from google.colab import drive
import pickle
import os

drive.mount('/content/drive')

# Step 2: Load Data
X_train_clean = np.load('/content/drive/My Drive/AI_Security_Project/X_train_clean.npy')
X_train_fgsm = np.load('/content/drive/My Drive/AI_Security_Project/X_train_fgsm.npy')
y_train_clean = np.load('/content/drive/My Drive/AI_Security_Project/y_train_clean.npy')

# Step 3: Quantum Tensor Purification (QTP)
class QTP(Model):
    def __init__(self, input_dim, latent_dim=32):
        super(QTP, self).__init__()
        self.encoder = tf.keras.Sequential([
            layers.Dense(128, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(latent_dim, activation='tanh')
        ])
        self.decoder = tf.keras.Sequential([
            layers.Dense(64, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(input_dim, activation='linear')
        ])

    def call(self, inputs):
        latent = self.encoder(inputs)
        latent_diffused = latent + tf.random.normal(tf.shape(latent), stddev=0.01, dtype=tf.float32)
        return self.decoder(latent_diffused)

qtp_file = '/content/drive/My Drive/AI_Security_Project/qtp_fgsm.pkl'
if os.path.exists(qtp_file):
    with open(qtp_file, 'rb') as f:
        qtp = pickle.load(f)
    print(f"Loaded pre-trained QTP from {qtp_file}")
else:
    qtp = QTP(X_train_clean.shape[1])
    scaler = StandardScaler()
    X_train_clean_scaled = scaler.fit_transform(X_train_clean)
    dataset = tf.data.Dataset.from_tensor_slices(X_train_clean_scaled).map(lambda x: tf.cast(x, tf.float32)).batch(40000).prefetch(tf.data.AUTOTUNE)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    for epoch in range(5):
        total_loss = 0
        for batch in dataset:
            with tf.GradientTape() as tape:
                recon = qtp(batch)
                loss = tf.reduce_mean(tf.square(recon - batch)) + 0.1 * tf.reduce_mean(tf.abs(recon))
            grads = tape.gradient(loss, qtp.trainable_variables)
            optimizer.apply_gradients(zip(grads, qtp.trainable_variables))
            total_loss += loss.numpy()
        print(f"QTP Epoch {epoch+1}, Loss: {total_loss:.4f}")
    with open(qtp_file, 'wb') as f:
        pickle.dump(qtp, f)
    print(f"QTP model saved to {qtp_file}")

# Step 4: Hyperdimensional Anomaly Isolator (HAI)
def hyperdimensional_encode(X, dim=10000):
    from sklearn.random_projection import GaussianRandomProjection
    projector = GaussianRandomProjection(n_components=dim, random_state=42)
    X_hd = projector.fit_transform(X)
    return X_hd / np.linalg.norm(X_hd, axis=1, keepdims=True)

X_clean_hd = hyperdimensional_encode(X_train_clean)
X_fgsm_hd = hyperdimensional_encode(X_train_fgsm)
np.save('/content/drive/My Drive/AI_Security_Project/hai_clean_fgsm.npy', X_clean_hd)
np.save('/content/drive/My Drive/AI_Security_Project/hai_poisoned_fgsm.npy', X_fgsm_hd)
print(f"HAI encodings saved to Drive: hai_clean_fgsm.npy, hai_poisoned_fgsm.npy")

# Cleanup
del X_train_clean, X_train_fgsm, y_train_clean, qtp, X_clean_hd, X_fgsm_hd
if 'scaler' in locals():
    del scaler
if 'dataset' in locals():
    del dataset
tf.keras.backend.clear_session()
print("QTP and HAI training completed, RAM cleared. Files safe in Google Drive.")