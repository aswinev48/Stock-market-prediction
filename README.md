# Stock-market-prediction
This project implements a deep learning solution using a **Stacked Long Short-Term Memory (LSTM) network** to predict the short-term trend of stock market data. The model classifies the expected movement of the next five candles (data points) into three categories: **Uptrend**, **Downtrend**, or **Neutral**.

## 🎯 Project Goal and Trend Definition

The primary objective is a **multi-class classification** task. For every time step, the model looks at 60 minutes of historical features and predicts the label for the immediate future.

The target labels are defined based on the requirement of the Roadlyft screening round:

| Class Label | Value | Definition (Next 5 Candles) |
| :--- | :--- | :--- |
| **Downtrend** | 0 | **All 5** subsequent candles have $\\text{Close} < \\text{Open}$ (i.e., all are red candles). |
| **Neutral** | 1 | Any mix of green and red candles. |
| **Uptrend** | 2 | **All 5** subsequent candles have $\\text{Close} > \\text{Open}$ (i.e., all are green candles). |

## 🛠️ Methodology and Model Architecture

### 1. Data Preparation and Preprocessing

* **Handling Missing Values:** Initial rows containing `NaN` values (due to insufficient historical data for calculating indicators like SMA 50 or RSI 14) are **dropped**.
* **Feature Scaling:** All 25 technical indicator features are scaled using **`MinMaxScaler(feature_range=(0, 1))`** to normalize the input data.
* **Sequence Creation:** The scaled data is converted into a 3D format: **(Samples, Timesteps, Features)**. A **60-timestep window** is used.
* **Label Encoding:** The integer trend labels (0, 1, 2) are converted into **one-hot encoded** vectors.
* **Class Imbalance:** **Class Weights** are used during training to address the dominance of the 'Neutral' class, improving the model's focus on 'Uptrend' and 'Downtrend' events.

### 2. Deep Learning Model (Stacked LSTM)

* **Type:** Stacked Long Short-Term Memory (LSTM).
* **Input Shape:** `(60, 25)`
* **Architecture:** **4 x LSTM Layers** (with 50-100 units each) followed by `Dropout` layers. The final layer is a `Dense(3)` with **`softmax`** activation.
* **Training Configuration:**
    * **Loss Function:** `categorical_crossentropy`.
    * **Optimizer:** `Adam`.
    * **Regularization:** `Dropout` and **Early Stopping** are implemented to prevent overfitting.

## ⚙️ Dependencies and Setup

This project requires the following libraries. All can be installed via `pip`:
```bash
pip install pandas numpy scikit-learn tensorflow matplotlib
```
### Execution Steps (Colab/Jupyter Notebook)
* **Upload Data:** Upload the provided large_32.csv file.

* **Environment Setup:** Run the imports cell to load all necessary libraries.

* **Run Code:** Execute all cells in the notebook (Stockmarket_Trend_Prediction (1).ipynb) sequentially, from Data Loading to Model Evaluation.

### 📝 Evaluation and Results
* **Metrics:** Model performance is assessed using Accuracy, Categorical Crossentropy Loss, a Classification Report (showing Precision, Recall, F1-Score per class), and a Confusion Matrix.

* **Model Persistence:** The final trained model is saved as a HDF5 file (stock_trend_lstm_model.h5) for easy deployment. """
