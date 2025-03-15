# Step 1: Install Libraries
!pip install tensorflow==2.15.0 numpy xgboost lightgbm sklearn torch

import numpy as np
import tensorflow as tf
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import accuracy_score
from google.colab import drive
import pickle
import os

drive.mount('/content/drive')

# Step 2: Load Data and Precomputed Models
X_train_clean = np.load('/content/drive/My Drive/AI_Security_Project/X_train_clean.npy')
X_train_fgsm = np.load('/content/drive/My Drive/AI_Security_Project/X_train_fgsm.npy')
y_train_clean = np.load('/content/drive/My Drive/AI_Security_Project/y_train_clean.npy')
X_test = np.load('/content/drive/My Drive/AI_Security_Project/X_test.npy')
y_test = np.load('/content/drive/My Drive/AI_Security_Project/y_test.npy')
X_clean_hd = np.load('/content/drive/My Drive/AI_Security_Project/hai_clean_fgsm.npy')
X_fgsm_hd = np.load('/content/drive/My Drive/AI_Security_Project/hai_poisoned_fgsm.npy')

with open('/content/drive/My Drive/AI_Security_Project/qtp_fgsm.pkl', 'rb') as f:
    qtp = pickle.load(f)
with open('/content/drive/My Drive/AI_Security_Project/nsfe_fgsm.pkl', 'rb') as f:
    nsfe = pickle.load(f)
with open('/content/drive/My Drive/AI_Security_Project/are_fgsm.pkl', 'rb') as f:
    are_model = pickle.load(f)
imlc = tf.keras.models.load_model('/content/drive/My Drive/AI_Security_Project/imlc_fgsm.h5', custom_objects={'IMLC': IMLC}) if os.path.exists('/content/drive/My Drive/AI_Security_Project/imlc_fgsm.h5') else IMLC()

# Step 3: FGSM Beast Corrector with Incremental Learning
def fgsm_beast_corrector(X_poisoned, y_clean, X_test, y_test, incremental=True):
    print("Running FGSM Beast Corrector...")

    # QTP Purification
    X_poisoned_scaled = scaler.transform(X_poisoned)
    X_qtp_corrected = qtp(X_poisoned_scaled)
    X_qtp_corrected = scaler.inverse_transform(X_qtp_corrected)

    # HAI Isolation
    X_fgsm_hd_corrected = X_fgsm_hd * (np.linalg.norm(X_clean_hd, axis=1, keepdims=True) / np.linalg.norm(X_fgsm_hd, axis=1, keepdims=True))
    X_hai_corrected = np.dot(X_fgsm_hd_corrected, projector.components_.T)  # Inverse projection

    # NSFE Refinement
    X_nsfe_corrected = nsfe(X_poisoned_scaled)
    X_nsfe_corrected = scaler.inverse_transform(X_nsfe_corrected)

    # Combine with IMLC Weights
    scores = np.vstack([X_qtp_corrected, X_hai_corrected, X_nsfe_corrected]).T
    weights = imlc(scores)
    X_corrected = tf.reduce_sum(scores * weights, axis=1).numpy()

    # ARE Retraining
    are_model.fit(X_corrected, y_clean)
    y_pred = are_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"FGSM Corrected Accuracy: {accuracy*100:.2f}%")

    # Incremental Update
    if incremental:
        dataset = tf.data.Dataset.from_tensor_slices((X_poisoned_scaled, X_corrected)).batch(40000).prefetch(tf.data.AUTOTUNE)
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        for epoch in range(3):
            total_loss = 0
            for batch_x, batch_y in dataset:
                with tf.GradientTape() as tape:
                    qtp_out = qtp(batch_x)
                    nsfe_out = nsfe(batch_x)
                    scores_batch = tf.stack([qtp_out, batch_x, nsfe_out], axis=1)  # HAI not re-trained, use original
                    weights_batch = imlc(scores_batch)
                    corrected_batch = tf.reduce_sum(scores_batch * weights_batch, axis=1)
                    loss = tf.reduce_mean(tf.square(corrected_batch - batch_y))
                grads = tape.gradient(loss, [qtp.trainable_variables, nsfe.trainable_variables, imlc.trainable_variables])
                optimizer.apply_gradients(zip(grads[0], qtp.trainable_variables))
                optimizer.apply_gradients(zip(grads[1], nsfe.trainable_variables))
                optimizer.apply_gradients(zip(grads[2], imlc.trainable_variables))
                total_loss += loss.numpy()
            print(f"Incremental Epoch {epoch+1}, Loss: {total_loss:.4f}")

    # Save Updated Models
    with open('/content/drive/My Drive/AI_Security_Project/qtp_fgsm.pkl', 'wb') as f:
        pickle.dump(qtp, f)
    with open('/content/drive/My Drive/AI_Security_Project/nsfe_fgsm.pkl', 'wb') as f:
        pickle.dump(nsfe, f)
    with open('/content/drive/My Drive/AI_Security_Project/are_fgsm.pkl', 'wb') as f:
        pickle.dump(are_model, f)
    imlc.save_weights('/content/drive/My Drive/AI_Security_Project/imlc_fgsm.h5')
    print("FGSM Beast Corrector models saved to Drive.")

    return X_corrected, are_model

# Step 4: Run Corrector
X_corrected_fgsm, are_model_fshm = fgsm_beast_corrector(X_train_fgsm, y_train_clean, X_test, y_test)

# Cleanup
del X_train_clean, X_train_fgsm, y_train_clean, X_test, y_test, X_clean_hd, X_fgsm_hd, qtp, nsfe, are_model, imlc, X_corrected_fgsm
tf.keras.backend.clear_session()
print("FGSM Beast Corrector completed, RAM cleared. Files safe in Google Drive.")