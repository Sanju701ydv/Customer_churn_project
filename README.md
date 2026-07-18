# Customer Churn Prediction

A machine learning web app that predicts whether a bank customer is likely to churn (leave the bank), built with an Artificial Neural Network (ANN) and deployed using Streamlit.

**Live App:** https://customerchurnprediction-3i8npxkbx84zekbpax3rapp.streamlit.app/

## Overview

Customer churn is a major cost driver for banks and subscription-based businesses. This project trains a neural network on historical customer data to predict the probability that a given customer will churn, based on features like credit score, age, balance, and account activity. The final app lets users enter customer details and instantly get a churn prediction in an interactive Streamlit interface.

## Dataset

- **Source:** `Churn_Modelling.csv` — a bank customer dataset with 10,000 records
- **Target variable:** `Exited` (1 = customer churned, 0 = customer stayed)
- **Key features:** Credit Score, Geography, Gender, Age, Tenure, Balance, Number of Products, Has Credit Card, Is Active Member, Estimated Salary

## Tools & Technologies

- **Python**
- **TensorFlow / Keras** – building and training the ANN
- **Scikit-learn** – data preprocessing (encoding, scaling)
- **NumPy** – lightweight inference engine for deployment (see note below)
- **Pandas** – data manipulation
- **Streamlit** – interactive web app for predictions
- **Jupyter Notebook** – model development and experimentation

## Steps

1. **Data Preprocessing**
   - Removed irrelevant columns (`RowNumber`, `CustomerId`, `Surname`)
   - Label-encoded `Gender` and one-hot encoded `Geography`
   - Split data into training (80%) and testing (20%) sets
   - Scaled features using `StandardScaler`

2. **Model Building**
   - Built a feed-forward ANN using Keras (`Dense` layers with ReLU activations and a sigmoid output layer for binary classification)
   - Compiled with the Adam optimizer and binary cross-entropy loss
   - Used early stopping to prevent overfitting

3. **Training & Evaluation**
   - Trained the model on the processed training set with validation on the test set
   - Tracked training progress with TensorBoard

4. **Deployment**
   - Saved the fitted preprocessing objects (encoders, scaler) with `pickle`
   - Extracted the trained model's weights and reimplemented the forward pass in pure NumPy (`model_weights.npz`), removing the TensorFlow runtime dependency for the deployed app
   - Built a Streamlit app (`app.py`) that loads these artifacts and lets users enter customer details to get a real-time churn prediction

> **Why NumPy instead of TensorFlow at deployment?** The model is trained with TensorFlow/Keras, but for the live app, its weights are exported and the forward pass (a few matrix multiplications and activation functions) is reimplemented in NumPy. This keeps the deployed app lightweight and avoids environment/dependency issues on hosting platforms, while producing numerically identical predictions.

## Results

- **Test Accuracy:** ~86%
- **Test Loss:** ~0.34 (binary cross-entropy)

The model reliably distinguishes between customers likely to stay and those likely to churn, making it a useful tool for proactive retention strategies.

## How to Run

1. **Clone the repository and install dependencies**
   ```bash
   git clone <repo-url>
   cd churn_project
   pip install -r requirements.txt
   ```

2. **Run the Streamlit app**
   ```bash
   streamlit run app.py
   ```

3. **Use the app**
   - Open the local URL shown in the terminal (usually `http://localhost:8501`)
   - Enter customer details (geography, gender, age, balance, etc.)
   - View the predicted churn probability and outcome instantly

## Project Structure

```
churn_project/
├── app.py                        # Streamlit web app (NumPy-based inference)
├── main.ipynb                    # Model development & training notebook (TensorFlow/Keras)
├── Churn_Modelling.csv           # Dataset
├── model_weights.npz             # Trained ANN weights (used by app.py)
├── scaler.pkl                    # Fitted StandardScaler
├── label_encoder_gender.pkl      # Fitted LabelEncoder for Gender
├── onehot_encoder_geo.pkl        # Fitted OneHotEncoder for Geography
├── requirements.txt              # Project dependencies
└── README.md                     # Project documentation
```
