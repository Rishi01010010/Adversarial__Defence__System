# Step 1: Install Libraries
!pip install scikit-learn numpy pandas tensorflow==2.15.0 adversarial-robustness-toolbox lightgbm
!pip install torch==2.5.1 -f https://download.pytorch.org/whl/torch_stable.html
!pip install torch-geometric

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from art.attacks.evasion import FastGradientMethod  # Placeholder, Backdoor doesn’t use ART here
from art.estimators.classification import SklearnClassifier
import torch
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from scipy.sparse import csr_matrix
import lightgbm as lgb
from google.colab import drive
import pickle
import os

drive.mount('/content/drive')

# Step 2: Load Data (Backdoor Only)
X_train_clean = np.load('/content/drive/My Drive/AI_Security_Project/X_train_clean.npy')
X_train_backdoor = np.load('/content/drive/My Drive/AI_Security_Project/X_train_backdoor.npy')
y_full = np.load('/content/drive/My Drive/AI_Security_Project/y_train_clean.npy')  # Note: Backdoor may have flipped labels, adjust if needed

# Step 3: Deep Denoising Autoencoder (DDAE)
def build_ddae(input_dim):
    inputs = layers.Input(shape=(input_dim,))
    enc = layers.Dense(128, activation='relu')(inputs)
    enc = layers.Dense(64, activation='relu')(enc)
    latent = layers.Dense(32, activation='relu')(enc)
    dec = layers.Dense(64, activation='relu')(latent)
    dec = layers.Dense(128, activation='relu')(dec)
    outputs = layers.Dense(input_dim, activation='linear')(dec)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse')
    return model

ddae = build_ddae(X_train_clean.shape[1])
dataset = tf.data.Dataset.from_tensor_slices((X_train_clean, X_train_clean)).batch(40000).prefetch(tf.data.AUTOTUNE)
ddae.fit(dataset, epochs=5, verbose=1)
with open('/content/drive/My Drive/AI_Security_Project/ddae_backdoor.pkl', 'wb') as f:
    pickle.dump(ddae, f)

def get_ddae_scores(X, model, batch_size=40000, output_file='ddae_scores.npy'):
    scores = []
    dataset = tf.data.Dataset.from_tensor_slices(X).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    for batch in dataset:
        recon = model.predict(batch, verbose=0)
        batch_scores = np.mean((batch.numpy() - recon) ** 2, axis=1)
        scores.append(batch_scores)
    scores = np.concatenate(scores)
    np.save(output_file, scores)
    return scores

# Step 4: Batch-Specific GNN
class GNN(torch.nn.Module):
    def __init__(self, in_channels):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(in_channels, 64)
        self.conv2 = GCNConv(64, 32)
        self.out = torch.nn.Linear(32, 1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        return self.out(x)

gnn = GNN(X_train_clean.shape[1])
optimizer = torch.optim.Adam(gnn.parameters(), lr=0.01)
criterion = torch.nn.MSELoss()
for epoch in range(5):
    gnn.train()
    batch_size = 40000
    total_loss = 0
    for i in range(0, len(X_train_clean), batch_size):
        X_batch = X_train_clean[i:i+batch_size]
        gnn_data_batch = build_gnn_data_batch(X_batch)
        optimizer.zero_grad()
        out = gnn(gnn_data_batch.x, gnn_data_batch.edge_index)
        loss = criterion(out, gnn_data_batch.x[:, :1])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Avg Batch Loss: {total_loss / (len(X_train_clean) // batch_size):.4f}")
torch.save(gnn.state_dict(), '/content/drive/My Drive/AI_Security_Project/gnn_backdoor.pth')

def build_gnn_data_batch(X_batch):
    from sklearn.neighbors import kneighbors_graph
    adj = kneighbors_graph(X_batch, n_neighbors=3, mode='connectivity', include_self=False)
    edge_index = torch.tensor(np.array(csr_matrix(adj).nonzero()), dtype=torch.long)
    return Data(x=torch.tensor(X_batch, dtype=torch.float), edge_index=edge_index)

def get_gnn_scores(X, gnn, batch_size=40000, output_file='gnn_scores.npy'):
    gnn.eval()
    scores = []
    for i in range(0, len(X), batch_size):
        X_batch = X[i:i+batch_size]
        gnn_data_batch = build_gnn_data_batch(X_batch)
        with torch.no_grad():
            batch_scores = gnn(gnn_data_batch.x, gnn_data_batch.edge_index).numpy().flatten()
        scores.append(batch_scores)
    scores = np.concatenate(scores)
    np.save(output_file, scores)
    return scores

# Step 5: Adversarial Feature Augmentation (AFA) with ART (Using FGSM as Placeholder for Backdoor)
X_subset, _, y_subset, _ = train_test_split(X_train_clean, y_full, train_size=10000, stratify=y_full, random_state=42)
scaler = StandardScaler()
X_subset_scaled = scaler.fit_transform(X_subset)

surrogate_model = LogisticRegression(max_iter=1000, solver='lbfgs')
surrogate_model.fit(X_subset_scaled, y_subset)

art_classifier = SklearnClassifier(model=surrogate_model)
fgsm_aug = FastGradientMethod(estimator=art_classifier, eps=0.1)  # Placeholder, Backdoor doesn’t need this
X_train_clean_scaled = scaler.transform(X_train_clean[:10000])
X_aug_fgsm = fgsm_aug.generate(X_train_clean_scaled)
X_augmented = np.vstack([X_train_clean[:10000], scaler.inverse_transform(X_aug_fgsm)])

# Step 6: Backdoor Beast Detector with Incremental Learning
class AdaptiveWeightLearner(tf.keras.Model):
    def __init__(self):
        super(AdaptiveWeightLearner, self).__init__()
        self.dense1 = layers.Dense(16, activation='relu')
        self.dense2 = layers.Dense(3, activation='softmax')

    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)

adaptive_learner = AdaptiveWeightLearner()
adaptive_learner.build(input_shape=(None, 3))
weight_file = '/content/drive/My Drive/AI_Security_Project/adaptive_weights_backdoor.h5'

def backdoor_beast_detector(X_clean, X_poisoned, incremental=True):
    print("Training Backdoor Beast Detector...")

    # DDAE Scores
    ddae_clean_scores = get_ddae_scores(X_clean, ddae, output_file='/content/ddae_clean_backdoor.npy')
    ddae_poisoned_scores = get_ddae_scores(X_poisoned, ddae, output_file='/content/ddae_poisoned_backdoor.npy')

    # GNN Scores
    gnn_clean_scores = get_gnn_scores(X_clean, gnn, output_file='/content/gnn_clean_backdoor.npy')
    gnn_poisoned_scores = get_gnn_scores(X_poisoned, gnn, output_file='/content/gnn_poisoned_backdoor.npy')

    # Isolation Forest
    iso_forest = IsolationForest(contamination=0.5, random_state=42, n_estimators=100)
    iso_forest.fit(np.vstack([X_clean[:50000], X_augmented]))
    iso_clean_scores = -iso_forest.decision_function(X_clean)
    iso_poisoned_scores = -iso_forest.decision_function(X_poisoned)
    np.save('/content/iso_clean_backdoor.npy', iso_clean_scores)
    np.save('/content/iso_poisoned_backdoor.npy', iso_poisoned_scores)

    # Ensemble with LightGBM - Incremental Training
    batch_size = 100000
    X_ensemble_clean = np.vstack([ddae_clean_scores, gnn_clean_scores, iso_clean_scores]).T
    X_ensemble_poisoned = np.vstack([ddae_poisoned_scores, gnn_poisoned_scores, iso_poisoned_scores]).T
    X_ensemble = np.vstack([X_ensemble_clean, X_ensemble_poisoned])
    y_ensemble = np.concatenate([np.zeros(len(X_clean)), np.ones(len(X_poisoned))])

    # Shuffle for balanced chunks
    indices = np.random.permutation(len(X_ensemble))
    X_ensemble_shuffled = X_ensemble[indices]
    y_ensemble_shuffled = y_ensemble[indices]

    # Load or initialize LightGBM
    lgb_file = '/content/drive/My Drive/AI_Security_Project/beast_detector_backdoor.pkl'
    if os.path.exists(lgb_file) and incremental:
        with open(lgb_file, 'rb') as f:
            lgb_model = pickle.load(f)
    else:
        lgb_model = lgb.LGBMClassifier(n_estimators=50, learning_rate=0.05)

    # Adaptive weighting
    if not os.path.exists(weight_file):
        adaptive_learner.compile(optimizer='adam', loss='categorical_crossentropy')
        weights = np.ones((len(X_ensemble), 3)) / 3
        adaptive_learner.fit(X_ensemble, weights, epochs=5, batch_size=10000, verbose=1)
    weights = adaptive_learner.predict(X_ensemble, batch_size=10000)
    adaptive_learner.save_weights(weight_file)

    # Incremental Training
    for i in range(0, len(X_ensemble_shuffled), batch_size):
        X_chunk = X_ensemble_shuffled[i:i+batch_size]
        y_chunk = y_ensemble_shuffled[i:i+batch_size]
        if i == 0 or not incremental or not os.path.exists(lgb_file):
            lgb_model.fit(X_chunk, y_chunk)
        else:
            lgb_model.fit(X_chunk, y_chunk, init_model=lgb_model)

    # Final Predictions with Unshuffled Data
    clean_pred = lgb_model.predict_proba(X_ensemble_clean)[:, 1]
    poisoned_pred = lgb_model.predict_proba(X_ensemble_poisoned)[:, 1]

    clean_anomaly_rate = np.mean(clean_pred > 0.5) * 100
    poisoned_anomaly_rate = np.mean(poisoned_pred > 0.5) * 100

    print("Backdoor Beast Detector Results:")
    print(f"Clean data anomaly rate: {clean_anomaly_rate:.2f}%")
    print(f"Poisoned data anomaly rate: {poisoned_anomaly_rate:.2f}%")
    print(f"Detection: {'Poisoned' if poisoned_anomaly_rate > 5 else 'Clean'}")
    print(f"Estimated poisoning percentage: {poisoned_anomaly_rate:.2f}%")

    auc = roc_auc_score(np.concatenate([np.zeros(len(X_clean)), np.ones(len(X_poisoned))]), np.concatenate([clean_pred, poisoned_pred]))
    print(f"AUC Score: {auc:.4f}")

    # Clean up
    for f in ['ddae_clean_backdoor.npy', 'ddae_poisoned_backdoor.npy', 'gnn_clean_backdoor.npy', 'gnn_poisoned_backdoor.npy', 'iso_clean_backdoor.npy', 'iso_poisoned_backdoor.npy']:
        os.remove(f'/content/{f}')

    return lgb_model, iso_forest, ddae, gnn

# Step 7: Train and Save Backdoor Detector
detector_backdoor, iso_backdoor, ddae_backdoor, gnn_backdoor = backdoor_beast_detector(X_train_clean, X_train_backdoor)
with open('/content/drive/My Drive/AI_Security_Project/beast_detector_backdoor.pkl', 'wb') as f:
    pickle.dump(detector_backdoor, f)
print("Backdoor Beast Detector saved to Drive.")