# Step 1: Setup and Imports
!pip install xgboost pandas numpy scikit-learn

import numpy as np
from sklearn.metrics import accuracy_score
import xgboost as xgb
from google.colab import drive

drive.mount('/content/drive')

# Step 2: Load Clean Data and Model 1
X_train_clean = np.load('/content/drive/My Drive/AI_Security_Project/X_train_clean.npy')
y_train_clean = np.load('/content/drive/My Drive/AI_Security_Project/y_train_clean.npy')
X_test = np.load('/content/drive/My Drive/AI_Security_Project/X_test.npy')
y_test = np.load('/content/drive/My Drive/AI_Security_Project/y_test.npy')
xgb_model = xgb.XGBClassifier()
xgb_model.load_model('/content/drive/My Drive/AI_Security_Project/xgb_model_1.json')

# Step 3: Backdoor Attack
def backdoor_attack(X, y, poison_rate=0.2):
    X_poisoned = X.copy()
    y_poisoned = y.copy()
    num_poison = int(len(X) * poison_rate)
    indices = np.random.choice(np.where(y == 0)[0], num_poison, replace=False)
    X_poisoned[indices, 263] = 1000  # MZ trigger
    X_poisoned[indices, 256] = 5000  # numstrings trigger
    y_poisoned[indices] = 1
    return X_poisoned, y_poisoned

print("Generating backdoor poisoned data...")
X_train_backdoor, y_train_backdoor = backdoor_attack(X_train_clean, y_train_clean, poison_rate=0.2)

# Step 4: Retrain Model 1 and Evaluate
print("Retraining Model 1 on backdoor poisoned data...")
xgb_model.fit(X_train_backdoor, y_train_backdoor)
y_pred = xgb_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on clean test data after Backdoor poisoning: {accuracy * 100:.2f}%")

# Step 5: Save Poisoned Dataset
np.save('/content/drive/My Drive/AI_Security_Project/X_train_backdoor.npy', X_train_backdoor)
np.save('/content/drive/My Drive/AI_Security_Project/y_train_backdoor.npy', y_train_backdoor)
print("Backdoor poisoned dataset saved to Drive.")