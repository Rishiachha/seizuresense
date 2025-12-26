# (Seizure Sense) Automated Seizure Detection Using EEG Signals  
(CHB-MIT Scalp EEG Dataset)

---

## Team Members

- Kotha Aaryan Reddy (SE22UCAM001)  
- Shilpa Sanivarapu (SE22UARI152)  
- Mamdi Srikar Reddy (SE22UARI090)  
- Kolanu Padmakanth Reddy (SE22UARI073)  
- Accha Srishanth Rishi (SE22UCSE007)

---

## Project Overview

This project implements an **automated epileptic seizure detection system** using multi-channel EEG signals from the **CHB-MIT Scalp EEG Dataset**.

The system:
- Preprocesses raw EEG signals
- Handles severe class imbalance
- Trains and evaluates:
  - CNN-LSTM model
  - Transformer-based model
  - Traditional ML classifiers (baseline)
- Visualizes training/testing behavior

---

## Folder / File Description

### 1️⃣ `Load_and_PreprocessCHB.py`

**Purpose:**  
Handles **complete EEG data loading and preprocessing**.

**What it does:**
- Reads raw `.edf` EEG files using MNE
- Selects **18 bipolar EEG channels**
- Resamples signals to **128 Hz**
- Splits EEG into **4-second non-overlapping windows**
- Labels each window as:
  - `1` → Seizure
  - `0` → Non-seizure
- Balances seizure and non-seizure samples
- Returns:
  - `X` → EEG windows `(samples, 18, 512)`
  - `y` → Binary labels

---

### 2️⃣ `Model_TrainingCHB.py`

**Purpose:**  
Defines and trains the **CNN-LSTM deep learning model**.

**Model architecture:**
- 1D Convolution layers → spatial feature extraction
- LSTM layers → temporal dependency learning
- Dropout + Batch Normalization → overfitting control

**Training details:**
- 5-fold cross-validation
- Binary cross-entropy loss
- Adam optimizer
- Returns:
  - Training history
  - Trained model
  - Test labels and predictions

---

### 3️⃣ `Model_Training_Transformers.py`

**Purpose:**  
Implements the **Transformer-based seizure detection model**.

**Model components:**
- Dense embedding layer
- Multi-Head Self-Attention
- Feed-forward layers
- Global Average Pooling

**Training details:**
- 5-fold cross-validation
- Attention-based temporal modeling
- Returns:
  - Training history
  - Trained model
  - Test labels and predictions

---

### 4️⃣ `ML_Classifiers_Training_ValidationCHB.py`

**Purpose:**  
Implements **traditional machine learning baselines**.

**Models trained:**
- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest

**Key points:**
- EEG windows are **flattened**
- 80–20 train-test split
- Evaluated using:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - MCC

---

### 5️⃣ `Visualization.py`

**Purpose:**  
Handles **model evaluation visualization and metrics**.

**Generates:**
- Training vs Testing Accuracy plots
- Training vs Testing Loss plots
- Confusion Matrix
- Final metrics:
  - Accuracy
  - F1-score
  - MCC

**Used for:**
- Overfitting analysis
- Performance comparison
- Report figures

---

### 6️⃣ `MainCHB.py`

**Purpose:**  
**Main execution file** of the project.

**What it controls:**
1. GPU detection
2. Dataset loading & normalization
3. CNN-LSTM training
4. Transformer training
5. Visualization generation
6. Traditional ML model training
7. Model saving

This is the **only file that needs to be run**.

---

## Execution Order (IMPORTANT)

Follow this order **implicitly** — do not run files individually.

MainCHB.py
├── Load_and_PreprocessCHB.py
├── Model_TrainingCHB.py
├── Model_Training_Transformers.py
├── Visualization.py
└── ML_Classifiers_Training_ValidationCHB.py


✅ **Only run `MainCHB.py`**

---

## Output Files (Generated Automatically)

### CNN-LSTM
- `cnn-lstm-training and testing.png`
- `cnn-lstm-training and testing-loss.png`

### Transformer
- `Transformer-training and testing.png`
- `Transformer-Training and testing-loss.png`

These plots are used in:
- Results & Discussion
- Overfitting analysis
- Performance comparison

---

## Key Observations from Plots

- CNN-LSTM shows **faster convergence** and **higher testing accuracy**
- Transformer shows **stable learning** but lower overall performance
- Mild overfitting observed due to limited seizure samples
- CNN-LSTM achieves better robustness under class imbalance

---

## Documents

- `PAPER.pdf` → Research paper 
- `REPORT.pdf` → Detailed academic project report

---

## Final Note

This project demonstrates that **deep learning models outperform traditional ML approaches** for EEG-based seizure detection, with the **CNN-LSTM model achieving the best balance of accuracy, recall, and MCC**, making it more suitable for clinical applications.

---
