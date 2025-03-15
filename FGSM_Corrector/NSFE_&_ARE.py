# Step 1: Install Libraries
!pip install tensorflow==2.15.0 numpy xgboost lightgbm sklearn torch

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from google.colab import drive
import pickle
import os

drive.mount('/content/drive')

# Step 2: Load Data and Precomputed Results
X_train_clean = np.load('/content/drive/My Drive/AI_Security_Project/X_train_clean.npy')
X_train_fgsm = np.load('/content/drive/My Drive/AI_Security_Project/X_train_fgsm.npy')
y_train_clean = np.load('/content/drive/My Drive/AI_Security_Project/y_train_clean.npy')
X_test = np.load('/content/drive/My Drive/AI_Security_Project/X_test.npy')
y_test = np.load('/content/drive/My Drive/AI_Security_Project/y_test.npy')
X_clean_hd = np.load('/content/drive/My Drive/AI_Security_Project/hai_clean_fgsm.npy')
X_fgsm_hd = np.load('/content/drive/My Drive/AI_Security_Project/hai_poisoned_fgsm.npy')

# Step 3: Neuro-Symbolic Feature Enforcer (NSFE)
class NSFE(Model):
    def __init__(self, input_dim):
        super(NSFE, self).__init__()
        self.dense = layers.Dense(input_dim, activation='relu')
        self.rules = tf.Variable(tf.random.uniform((input_dim, 2)), trainable=True)  # Min/max bounds per feature

    def call(self, inputs):
        x = self.dense(inputs)
        return tf.clip_by_value(x, self.rules[:, 0], self.rules[:, 1])

nsfe_file = '/content/drive/My Drive/AI_Security_Project/nsfe_fgsm.pkl'
if os.path.exists(nsfe_file):
    with open(nsfe_file, 'rb') as f:
        nsfe = pickle.load(f)
else:
    nsfe = NSFE(X_train_clean.shape[1])
    scaler = StandardScaler()
    X_train_clean_scaled = scaler.fit_transform(X_train_clean)
    dataset = tf.data.Dataset.from_tensor_slices(X_train_clean_scaled).batch(40000).prefetch(tf.data.AUTOTUNE)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    for epoch in range(5):
        total_loss = 0
        for batch in dataset:
            with tf.GradientTape() as tape:
                corrected = nsfe(batch)
                loss = tf.reduce_mean(tf.square(corrected - batch))
            grads = tape.gradient(loss, nsfe.trainable_variables)
            optimizer.apply_gradients(zip(grads, nsfe.trainable_variables))
            total_loss += loss.numpy()
        print(f"NSFE Epoch {epoch+1}, Loss: {total_loss:.4f}")
    with open(nsfe_file, 'wb') as f:
        pickle.dump(nsfe, f)
    print(f"NSFE model saved to {nsfe_file}")

# Step 4: Adversarial Retraining Engine (ARE)
def adversarial_retrain(X_corrected, y, X_test, y_test):
    model = xgb.XGBClassifier()
    model.fit(X_corrected, y)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"ARE Initial Accuracy: {accuracy*100:.2f}%")
    return model

are_file = '/content/drive/My Drive/AI_Security_Project/are_fgsm.pkl'
if os.path.exists(are_file):
    with open(are_file, 'rb') as f:
        are_model = pickle.load(f)
else:
    are_model = adversarial_retrain(X_train_clean, y_train_clean, X_test, y_test)
    with open(are_file, 'wb') as f:
        pickle.dump(are_model, f)

# Step 5: Incremental Meta-Learning Core (IMLC)
class IMLC(Model):
    def __init__(self):
        super(IMLC, self).__init__()
        self.meta = layers.Dense(32, activation='relu')
        self.out = layers.Dense(3, activation='softmax')  # Weights for purification components

    def call(self, inputs):
        x = self.meta(inputs)
        return self.out(x)

imlc_file = '/content/drive/My Drive/AI_Security_Project/imlc_fgsm.h5'
if os.path.exists(imlc_file):
    imlc = IMLC()
    imlc.load_weights(imlc_file)
else:
    imlc = IMLC()

# Cleanup
del X_train_clean, X_train_fgsm, y_train_clean, X_test, y_test, X_clean_hd, X_fgsm_hd, nsfe, are_model
if 'scaler' in locals():
    del scaler
if 'dataset' in locals():
    del dataset
tf.keras.backend.clear_session()
print("NSFE, ARE, and IMLC setup completed, RAM cleared. Files safe in Google Drive.")