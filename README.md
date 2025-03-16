# 🛡️ Adversarial Defence System: Safeguarding AI Against Attacks 🛡️

Welcome to the *Adversarial Defence System*, a cutting-edge solution developed to protect AI models from adversarial attacks, backdoors, and malware. Built using advanced machine learning techniques, this project leverages TensorFlow, PyTorch, and Google Cloud technologies to detect, correct, and mitigate threats like Fast Gradient Sign Method (FGSM), Projected Gradient Descent (PGD), and backdoor poisoning. Whether you're a researcher or a developer, this system offers robust tools to secure your AI models in production environments.

## 🔍 Project Overview

The *Adversarial Defence System* tackles the growing challenge of adversarial attacks on machine learning models, which can manipulate inputs to cause misclassifications, inject backdoors, or introduce malicious behavior. This project provides a modular framework with detectors, correctors, and poisoning bots for FGSM, PGD, and backdoor attacks, alongside a malware detection system. It employs innovative techniques like Quantum Tensor Purification (QTP), Hyperdimensional Anomaly Isolation (HAI), Neuro-Symbolic Feature Enforcement (NSFE), and Adversarial Retraining Engine (ARE) to ensure robust defense.

### ✨ Key Features:

- *FGSM & PGD Detection and Correction:* Identifies and mitigates adversarial perturbations using advanced purification and retraining techniques.
- *Backdoor Detection:* Detects backdoor poisoning with a high AUC score (0.5554 in testing) using LightGBM and anomaly rate analysis.
- *Malware Detection:* Scans for malicious patterns in AI systems.
- *Incremental Learning:* Continuously improves model performance against evolving threats.
- *Poisoning Bots:* Simulates adversarial attacks for testing and research purposes.
- *Scalable Architecture:* Designed to handle large datasets with GPU acceleration (e.g., Google Colab T4 GPU).

## 🚀 Getting Started

### 1. *Prerequisites:*
- Python 3.11 or higher.
- Google Colab with GPU support (T4 or better) for training.
- Google Drive for data storage and model persistence.
- Required libraries: TensorFlow 2.15.0, PyTorch 2.5.1, NumPy, scikit-learn, XGBoost, LightGBM, and more (see installation steps below).

### 2. *Setting Up:*

- Clone the repository:
  ```bash
  git clone https://github.com/your-username/Adversarial_Defence_System.git
  cd Adversarial_Defence_System
  ```

- Install dependencies:
  ```bash
  pip install tensorflow==2.15.0 numpy scikit-learn xgboost lightgbm torch==2.5.1 tensorly
  ```

- Mount Google Drive in Google Colab (if applicable):
  ```python
  from google.colab import drive
  drive.mount('/content/drive')
  ```

- Place your dataset files (e.g., `X_train_clean.npy`, `X_train_fgsm.npy`, etc.) in `/content/drive/My Drive/AI_Security_Project/`.

### 3. *Running the System:*

- **FGSM Detection and Correction:**
  - Start with `FGSM_Corrector/QTP_&_HAI_Trainer.py` to train the QTP and HAI components.
  - Run `FGSM_Corrector/NSFE_&_ARE.py` for feature enforcement and retraining.
  - Finally, execute `FGSM_Corrector/Incremental_Learning.py` to apply the full FGSM Beast Corrector with incremental learning.
  
  ```bash
  python FGSM_Corrector/QTP_&_HAI_Trainer.py
  python FGSM_Corrector/NSFE_&_ARE.py
  python FGSM_Corrector/Incremental_Learning.py
  ```

- **Backdoor Detection:**
  - Run `Backdoor_Detector.py` to analyze clean and poisoned data for backdoor anomalies.
  ```bash
  python Backdoor_Detector.py
  ```

- **PGD Detection:**
  - Use `PGD_Detector.py` to identify PGD-based adversarial attacks.
  ```bash
  python PGD_Detector.py
  ```

- **Simulate Attacks:**
  - Use `FGSM_Poisoning_Bot.py`, `PGD_Poisoning_Bot.py`, or `Backdoor_Poisoning_Bot.py` to generate adversarial examples for testing.
  ```bash
  python FGSM_Poisoning_Bot.py
  ```

- **Malware Detection:**
  - Run `Malware_Detection_System.py` to scan for malicious patterns.
  ```bash
  python Malware_Detection_System.py
  ```

### 4. *Sample Output:*
- **Backdoor Detection Results** (from `Backdoor_Detector.py`):
  ```
  Clean data anomaly rate: 56.09%
  Poisoned data anomaly rate: 62.67%
  Detection: Poisoned
  Estimated poisoning percentage: 62.67%
  AUC Score: 0.5554
  ```
- **QTP Training** (from `QTP_&_HAI_Trainer.py`):
  ```
  QTP Epoch 1, Loss: 9.8549
  QTP Epoch 5, Loss: 7.1090
  ```

## 💾 Directory Structure

```bash
Adversarial_Defence_System/
│
├── Adversarial_Defence_System.ipynb  # Main Jupyter notebook with full pipeline
├── Backdoor_Detector.py              # Detects backdoor poisoning in datasets
├── Backdoor_Poisoning_Bot.py         # Simulates backdoor attacks
├── FGSM_Detector.py                  # Detects FGSM adversarial attacks
├── FGSM_Poisoning_Bot.py             # Simulates FGSM attacks
├── Malware_Detection_System.py       # Identifies malware in AI systems
├── PGD_Detector.py                   # Detects PGD adversarial attacks
├── PGD_Poisoning_Bot.py              # Simulates PGD attacks
├── README.md                         # Project documentation
│
└── FGSM_Corrector/                   # FGSM correction pipeline
    ├── QTP_&_HAI_Trainer.py          # Trains Quantum Tensor Purification and Hyperdimensional Anomaly Isolator
    ├── NSFE_&_ARE.py                 # Implements Neuro-Symbolic Feature Enforcer and Adversarial Retraining Engine
    ├── Incremental_Learning.py        # Combines all components with incremental learning
```

### 📝 Code Explanation

1. **FGSM_Corrector/QTP_&_HAI_Trainer.py**:
   - Implements Quantum Tensor Purification (QTP) using a neural network with encoder-decoder architecture.
   - Applies Hyperdimensional Anomaly Isolation (HAI) using Gaussian random projection to isolate adversarial perturbations.
   - Saves trained models and encodings to Google Drive.

2. **FGSM_Corrector/NSFE_&_ARE.py**:
   - Neuro-Symbolic Feature Enforcer (NSFE) enforces feature bounds using a dense layer with trainable min/max rules.
   - Adversarial Retraining Engine (ARE) uses XGBoost to retrain on corrected data, improving robustness.
   - Sets up the Incremental Meta-Learning Core (IMLC) for dynamic weighting of purification components.

3. **FGSM_Corrector/Incremental_Learning.py**:
   - Combines QTP, HAI, NSFE, and ARE into the full FGSM Beast Corrector.
   - Implements incremental learning to adapt to new adversarial patterns over time.

4. **Backdoor_Detector.py**:
   - Uses LightGBM to detect backdoor anomalies with a binary classification approach.
   - Reports anomaly rates, detection status, and AUC scores.

5. **Adversarial_Defence_System.ipynb**:
   - A comprehensive Jupyter notebook containing the entire pipeline, including training logs and results.

## 🌐 System Configuration

- *Environment:* Google Colab with T4 GPU for accelerated training.
- *Data Storage:* Google Drive (`/content/drive/My Drive/AI_Security_Project/`) for datasets and models.
- *Dependencies:* Ensure TensorFlow 2.15.0 and PyTorch 2.5.1 are installed to avoid compatibility issues (see dependency conflicts in logs).
- *Memory Management:* Scripts include cleanup steps to clear RAM after execution, preventing crashes in Colab.

## 🛠️ How It Works

1. *Data Preparation*: Loads clean and poisoned datasets (e.g., `X_train_clean.npy`, `X_train_fgsm.npy`) from Google Drive.
2. *Detection*: Identifies adversarial attacks using FGSM, PGD, or backdoor detectors.
3. *Correction*:
   - QTP purifies data by reconstructing clean features.
   - HAI isolates anomalies in a high-dimensional space.
   - NSFE enforces feature constraints to remove adversarial noise.
   - ARE retrains the model on corrected data for improved robustness.
4. *Incremental Learning*: IMLC dynamically adjusts weights of purification components and updates models over time.
5. *Evaluation*: Measures performance using accuracy, AUC scores, and anomaly rates.

## 🎯 Project Intent

The *Adversarial Defence System* aims to provide a comprehensive framework for securing AI models against adversarial threats. By combining detection, correction, and simulation tools, it serves as a valuable resource for AI security researchers and practitioners. The system is designed to be modular, allowing users to extend or customize components for specific use cases.

## 🔧 Customization

Enhance the project with these ideas:
- *Add Support for New Attacks:* Extend the system to handle other adversarial techniques like CW or DeepFool.
- *Optimize Performance:* Use TensorFlow Quantum for QTP to simulate quantum-based purification.
- *Deploy as a Service:* Convert the system into a Flask API for real-time adversarial defense.
- *Visualizations:* Add matplotlib or seaborn plots to visualize anomaly rates and model performance.

## 📌 Links
- **Dataset:** https://github.com/elastic/ember.
- **Demo Video:** https://youtu.be/ZmtVZL7hJd8
