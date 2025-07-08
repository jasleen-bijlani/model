# smart_neutralization.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_excel("C:/Intern/Deliverables/pH model/pH model.xlsx")

# Constants
pKa1 = 3.13
pKa2 = 4.76
target_pH = 5.5
MW_NaOH = 40
dilution_factor = 11
MW_citric_acid = 192.12

# Theoretical model
def theoretical_naoh(pH_initial, quantity_kl):
    mass_citric_acid = quantity_kl
    n_citric = (mass_citric_acid * 1000) / MW_citric_acid

    alpha_i = 1 / (1 + 10**(pKa1 - pH_initial)) + 1 / (1 + 10**(pKa2 - pH_initial))
    alpha_f = 1 / (1 + 10**(pKa1 - target_pH)) + 1 / (1 + 10**(pKa2 - target_pH))

    delta_moles = n_citric * (alpha_f - alpha_i)
    return delta_moles * MW_NaOH / 1000 * dilution_factor

# Prepare training data
df["Theoretical NaOH (kg)"] = df.apply(
    lambda row: theoretical_naoh(row["pH before"], row["Quantity(kL)"]),
    axis=1
)
df["Residual (kg)"] = df["NaOH solution added(kg)"] - df["Theoretical NaOH (kg)"]

X = df[["Quantity(kL)", "pH before", "pH after"]]
y = df["Residual (kg)"]

X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit models
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

edge_train_idx = X_train[(X_train["pH before"] < 3.3) | (X_train["pH before"] > 4)].index
lin_reg = None
if not edge_train_idx.empty:
    lin_reg = LinearRegression()
    lin_reg.fit(X_train.loc[edge_train_idx, ["pH before"]], y_train.loc[edge_train_idx])

# Final prediction function
def hybrid_predict(pH_before, quantity_kl):
    theoretical_val = theoretical_naoh(pH_before, quantity_kl)
    if 3.3 <= pH_before <= 4:
        residual_pred = rf.predict([[quantity_kl, pH_before, target_pH]])[0]
        return round(theoretical_val + residual_pred, 2)
    else:
        if lin_reg:
            bias_pred = lin_reg.predict([[pH_before]])[0]
            return round(theoretical_val + bias_pred, 2)
        else:
            fallback_residual = y_train.mean()
            return round(theoretical_val + fallback_residual, 2)

# ---------------- STREAMLIT APP ----------------

st.title("Smart Neutralization Model")
st.write("Enter batch details to estimate NaOH requirement.")

# Inputs
pH_before = st.number_input("Initial pH", min_value=2.8, max_value=4.5, value=3.5, step=0.01)
quantity = st.selectbox("Batch Quantity (kL)", options=[10.0, 3.0])

# Predict
if st.button("Predict NaOH Required"):
    final_pred = hybrid_predict(pH_before, quantity)
    st.success(f"Estimated NaOH Required: **{final_pred} kg**")
