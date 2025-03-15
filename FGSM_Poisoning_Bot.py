import numpy as np
from sklearn.metrics import accuracy_score
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras import layers, models
from google.colab import drive

drive.mount('/content/drive')

# Step 2: Load Clean Data and Model 1
X_train_clean = np.load('/content/drive/My Drive/AI_Security_Project/X_train_clean.npy')
y_train_clean = np.load('/content/drive/My Drive/AI_Security_Project/y_train_clean.npy')
X_test = np.load('/content/drive/My Drive/AI_Security_Project/X_test.npy')
y_test = np.load('/content/drive/My Drive/AI_Security_Project/y_test.npy')
xgb_model = xgb.XGBClassifier()
xgb_model.load_model('/content/drive/My Drive/AI_Security_Project/xgb_model_1.json')

# Step 3: Surrogate Neural Network
def build_surrogate_model(input_dim):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

X_train_normalized = (X_train_clean - X_train_clean.mean(axis=0)) / (X_train_clean.std(axis=0) + 1e-8)
surrogate_model = build_surrogate_model(X_train_normalized.shape[1])
surrogate_model.fit(X_train_normalized, y_train_clean, epochs=5, batch_size=32, verbose=1)

# Step 4: FGSM Attack
def fgsm_attack(X, y, model, epsilon=0.05):
    X_tensor = tf.convert_to_tensor(X, dtype=tf.float32)
    y_tensor = tf.convert_to_tensor(y, dtype=tf.float32)[:, tf.newaxis]
    with tf.GradientTape() as tape:
        tape.watch(X_tensor)
        predictions = model(X_tensor)
        loss = tf.keras.losses.binary_crossentropy(y_tensor, predictions)
    gradient = tape.gradient(loss, X_tensor)
    perturbation = epsilon * tf.sign(gradient)
    X_poisoned = X_tensor + perturbation
    return X_poisoned.numpy()

print("Generating FGSM poisoned data...")
X_train_fgsm = fgsm_attack(X_train_normalized, y_train_clean, surrogate_model, epsilon=0.05)

# Step 5: Retrain Model 1 and Evaluate
print("Retraining Model 1 on FGSM poisoned data...")
X_train_fgsm_denorm = X_train_fgsm * (X_train_clean.std(axis=0) + 1e-8) + X_train_clean.mean(axis=0)
xgb_model.fit(X_train_fgsm_denorm, y_train_clean)
y_pred = xgb_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on clean test data after FGSM poisoning: {accuracy * 100:.2f}%")

# Step 6: Save Poisoned Dataset
np.save('/content/drive/My Drive/AI_Security_Project/X_train_fgsm.npy', X_train_fgsm_denorm)
print("FGSM poisoned dataset saved to Drive.")