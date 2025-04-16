import os
import warnings
import streamlit as st
import yfinance as yf
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppresses info and warnings

# Suppress Keras warnings
warnings.filterwarnings("ignore", category=UserWarning, module="keras")

# Suppress YFinance download warnings
warnings.filterwarnings("ignore", category=UserWarning, module="yfinance")

# Define assets
assets = {
    "Gold": "GC=F",
    "Silver": "SI=F",
    "Crude Oil (WTI)": "CL=F",
    "Brent Oil": "BZ=F",
    "Natural Gas": "NG=F",
    "Copper": "HG=F",
    "Corn": "ZC=F",
    "Wheat": "ZW=F",
    "Soybeans": "ZS=F",
    "Coffee": "KC=F",
    "Cocoa": "CC=F",
    "Live Cattle": "LE=F",
    "Lean Hogs": "HE=F",
    "Dollar Index": "DX-Y.NYB",
    "Dow Jones Industrial Average": "^DJI",
    "Euro/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "USD/JPY": "JPY=X",
    "USD/CHF": "CHF=X",
    "AUD/USD": "AUDUSD=X"
}

st.title("📈 Asset Lag & Granger Causality Analysis")

# Select features and target
selected_features = st.multiselect(
    "Select feature assets (X variables)", list(assets.keys()), default=["Gold", "Silver"]
)
selected_target = st.selectbox(
    "Select target asset (Y variable)", list(assets.keys()), index=list(assets.keys()).index("Live Cattle")
)
ticker = assets[selected_target]

# Download feature data
if selected_features:
    if st.button("Download Feature Data"):
        feature_tickers = [assets[asset] for asset in selected_features]
        end_date = datetime.today()
        start_date = end_date - timedelta(weeks=156)
        df_weekly = yf.download(
            feature_tickers, start=start_date.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d'), interval="1wk", auto_adjust=False
        )["Close"]
        df_weekly.rename(columns={v: k for k, v in assets.items()}, inplace=True)
        if isinstance(df_weekly.columns, pd.MultiIndex):
            df_weekly.columns = df_weekly.columns.droplevel(0)
        df_weekly = df_weekly.ffill().bfill().round(2)
        df_weekly.index = df_weekly.index.date
        st.session_state.df_final = df_weekly
        st.dataframe(df_weekly.tail())
        st.success("✅ Feature data downloaded.")

# Download target data
if 'df_final' in st.session_state:
    if st.button("Download Target Data"):
        end_date = datetime.today()
        start_date = end_date - timedelta(weeks=104)
        df_target = yf.download(
            ticker, start=start_date.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d'), interval="1wk", auto_adjust=False
        )[["Close"]]
        df_target.index = pd.to_datetime(df_target.index).date
        df_target.columns = ["target"]
        df_target = df_target.round(2).dropna()
        st.session_state.df_target = df_target
        st.dataframe(df_target.tail())
        st.success("✅ Target data downloaded.")

# Create lag dataframes
if 'df_final' in st.session_state and 'df_target' in st.session_state:
    if st.button("Create Lag DataFrames"):
        df_final = st.session_state.df_final.sort_index(ascending=True)
        window_size = 104
        step_size = 13
        df_lag_12m = df_final.iloc[:window_size].copy()
        df_lag_12m.columns = [f"{col}_lag_12m" for col in df_lag_12m.columns]
        df_lag_9m = df_final.iloc[step_size:step_size + window_size].copy()
        df_lag_9m.columns = [f"{col}_lag_9m" for col in df_lag_9m.columns]
        df_lag_6m = df_final.iloc[2 * step_size:2 * step_size + window_size].copy()
        df_lag_6m.columns = [f"{col}_lag_6m" for col in df_lag_6m.columns]
        df_lag_3m = df_final.iloc[3 * step_size:3 * step_size + window_size].copy()
        df_lag_3m.columns = [f"{col}_lag_3m" for col in df_lag_3m.columns]
        st.session_state.df_lag_12m = df_lag_12m
        st.session_state.df_lag_9m = df_lag_9m
        st.session_state.df_lag_6m = df_lag_6m
        st.session_state.df_lag_3m = df_lag_3m
        st.write("✅ 12-month lag")
        st.dataframe(df_lag_12m.head())
        st.write("✅ 9-month lag")
        st.dataframe(df_lag_9m.head())
        st.write("✅ 6-month lag")
        st.dataframe(df_lag_6m.head())
        st.write("✅ 3-month lag")
        st.dataframe(df_lag_3m.head())
        st.success("✅ All lag DataFrames created.")

# Merge
if all(key in st.session_state for key in ['df_lag_3m', 'df_lag_6m', 'df_lag_9m', 'df_lag_12m', 'df_target']):
    if st.button("Merge DataFrames"):
        df_merged = st.session_state.df_target.reset_index(drop=True) \
            .merge(st.session_state.df_lag_3m.reset_index(drop=True), left_index=True, right_index=True) \
            .merge(st.session_state.df_lag_6m.reset_index(drop=True), left_index=True, right_index=True) \
            .merge(st.session_state.df_lag_9m.reset_index(drop=True), left_index=True, right_index=True) \
            .merge(st.session_state.df_lag_12m.reset_index(drop=True), left_index=True, right_index=True)
        st.session_state.df_merged = df_merged
        st.dataframe(df_merged.head())
        st.success("✅ Merged DataFrame ready.")

# Correlations
if 'df_merged' in st.session_state:
    # Display a comment explaining what the user is about to do
    st.markdown("### Let's see what the lagged features are correlated!")
    if st.button("Run Correlation Analysis"):
        corr_matrix = st.session_state.df_merged.corr()
        correlations = corr_matrix["target"].drop("target").sort_values(ascending=False)
        threshold = 0.65
        filtered_cols = correlations[correlations.abs() > threshold].index.tolist()
        if not filtered_cols:
            st.warning("❗ No strong correlations found. Choose another target.")
        else:
            filtered_cols.append("target")
            df_correlations = st.session_state.df_merged[filtered_cols]
            st.session_state.df_correlations = df_correlations
            st.dataframe(df_correlations.head())
            st.success("✅ Correlation filtering done.")

# Granger Test
if 'df_correlations' in st.session_state:
    if st.button("Run Granger Test and Filter"):
        df_correlations = st.session_state.df_correlations.copy()
        df_correlations.dropna(inplace=True)
        X_columns = [col for col in df_correlations.columns if "target" not in col]
        max_lags = 5
        granger_results = {}
        warnings.simplefilter(action="ignore", category=FutureWarning)

        for col in X_columns:
            test_result = grangercausalitytests(df_correlations[[col, "target"]], max_lags, verbose=False)
            p_vals = [round(test_result[i + 1][0]['ssr_ftest'][1], 4) for i in range(max_lags)]
            granger_results[col] = p_vals

        granger_df = pd.DataFrame(granger_results, index=[f"Lag {i+1}" for i in range(max_lags)])
        st.session_state.granger_df = granger_df

        threshold = 0.1  # Testing threshold
        filtered = granger_df.applymap(lambda x: x if x < threshold else None).dropna(how="all", axis=1)
        filtered = filtered.reset_index().melt(id_vars="index", var_name="Feature", value_name="Granger_p_value")
        filtered = filtered.dropna().reset_index(drop=True)
        filtered.rename(columns={"index": "Granger_Lag_Amount"}, inplace=True)
        filtered["Granger_Lag_Amount"] = filtered["Granger_Lag_Amount"].astype(str).str.extract("(\d+)").astype(int)

        best_lags = filtered.loc[filtered.groupby("Feature")["Granger_p_value"].idxmin()]
        expanded = []
        for _, row in best_lags.iterrows():
            for lag in range(1, row["Granger_Lag_Amount"] + 1):
                expanded.append({
                    "Feature": row["Feature"],
                    "Granger_Lag_Amount": lag,
                    "Granger_p_value": row["Granger_p_value"] if lag == row["Granger_Lag_Amount"] else None
                })

        if len(expanded) == 0:
            st.warning("❗ No features passed Granger threshold. Please try a different feature/target combination.")
        else:
            final_filtered = pd.DataFrame(expanded).dropna(subset=["Granger_p_value"]).reset_index(drop=True)
            st.session_state.filtered_granger_df = final_filtered
            st.dataframe(final_filtered)
            st.success("✅ Filtered Granger results saved.")

# Visual plots
if 'filtered_granger_df' in st.session_state and 'df_correlations' in st.session_state:
    # Display a comment explaining what the user is about to do
    st.markdown("### Display the correlated and granger filtered features to target ")
    if st.button("Show Correlation Visuals"):
        feature_list = st.session_state.filtered_granger_df["Feature"].unique().tolist()
        df_correlations = st.session_state.df_correlations
        target_normalized = (df_correlations["target"] - df_correlations["target"].min()) / \
                            (df_correlations["target"].max() - df_correlations["target"].min())
        for feature in feature_list:
            if feature in df_correlations.columns:
                feature_normalized = (df_correlations[feature] - df_correlations[feature].min()) / \
                                     (df_correlations[feature].max() - df_correlations[feature].min())
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(df_correlations.index, feature_normalized, label=f"{feature} (Normalized)", color='orange')
                ax.plot(df_correlations.index, target_normalized, label=f"{selected_target} (Normalized)", linestyle='--', color='black')
                ax.set_xlabel("Index")
                ax.set_ylabel("Normalized Values")
                ax.set_title(f"{feature} vs. {selected_target}")
                ax.grid(True)
                ax.legend()
                st.pyplot(fig)
        st.success("✅ Visual plots displayed.")

# Add new features button (all working & saved)
if 'filtered_granger_df' in st.session_state and 'df_correlations' in st.session_state:
    if st.button("Calculate New Features"):
        filtered_granger_df = st.session_state.filtered_granger_df
        df_correlations = st.session_state.df_correlations.copy()
        selected_features = filtered_granger_df["Feature"].unique().tolist()
        df_lagged = df_correlations[selected_features].copy()
        df_lagged["target"] = df_correlations["target"]
        for _, row in filtered_granger_df.iterrows():
            feature = row["Feature"]
            max_lag = int(row["Granger_Lag_Amount"])
            if feature in df_lagged.columns:
                for lag in range(1, max_lag + 1):
                    df_lagged[f"{feature}_shifted_{lag}"] = df_lagged[feature].shift(lag)
        df_lagged = df_lagged.ffill().bfill()
        st.session_state.df_lagged = df_lagged
        st.subheader("✅ New Lag Features Created (Head Only)")
        st.dataframe(df_lagged.head())
        st.success("✅ All new lagged features ready.")



# Scaler check block with button, saving to session_state
if 'df_lagged' in st.session_state:
    if st.button("Run Scaler Check"):
        df_ml = st.session_state.df_lagged.copy()
        scalers = {
            "StandardScaler": StandardScaler(),
            "MinMaxScaler": MinMaxScaler(),
            "RobustScaler": RobustScaler(),
            "MaxAbsScaler": MaxAbsScaler()
        }
        results = {}
        for name, scaler in scalers.items():
            df_scaled = df_ml.copy()
            df_scaled.iloc[:, :] = scaler.fit_transform(df_scaled)
            X = df_scaled.drop(columns=["target"])
            y = df_scaled["target"]
            X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, shuffle=False)
            results[name] = {
                "X_train_mean": X_train.mean().mean(),
                "X_train_std": X_train.std().mean(),
                "y_train_mean": y_train.mean(),
                "y_train_std": y_train.std()
            }
        results_df = pd.DataFrame(results).T
        best_scaler_name = results_df['y_train_std'].idxmin()
        st.session_state.best_scaler_name = best_scaler_name
        st.session_state.best_scaler = scalers[best_scaler_name]
        st.subheader("Scaler Check Results:")
        st.dataframe(results_df)
        st.success(f"✅ Best scaler chosen: {best_scaler_name}")

import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler

# Ensure that the scaling and data preparation happens first
if 'df_lagged' in st.session_state and 'best_scaler' in st.session_state:
    if st.button("Scale Data & Prepare Train/Test"):
        df_lagged = st.session_state.df_lagged.copy()
        scaler = st.session_state.best_scaler

        feature_scaler = scaler
        target_scaler = scaler

        # Fit the scaler to the feature data and target data
        X_scaled = feature_scaler.fit_transform(df_lagged.drop(columns=["target"]))
        y_scaled = target_scaler.fit_transform(df_lagged[["target"]])

        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, shuffle=False)

        # Save the scaled data to session_state for later use
        st.session_state.X_train = X_train
        st.session_state.X_test = X_test
        st.session_state.y_train = y_train
        st.session_state.y_test = y_test
        st.session_state.target_scaler = target_scaler

        st.success("✅ Data scaled and train/test sets prepared.")


# Now define the model and training button
if all(key in st.session_state for key in ['X_train', 'X_test', 'y_train', 'y_test']):
    if st.button("Train the Model"):
        # Define the Neural Network (MLP)
        model = keras.Sequential([
            layers.Dense(128, activation="relu", input_shape=(st.session_state.X_train.shape[1],)),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(64, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(32, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(1)  # Output layer (regression task)
        ])

        # Compile the model
        model.compile(optimizer="adam", loss="mse", metrics=["mae"])

        # Create an empty placeholder for the progress bar
        progress_bar = st.empty()

        # Define the progress bar callback for Streamlit
        class ProgressBar(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                progress = (epoch + 1) / self.params["epochs"]
                progress_bar.progress(progress)  # Update Streamlit progress bar horizontally

        # Train the model
        history = model.fit(
            st.session_state.X_train, 
            st.session_state.y_train, 
            epochs=100, 
            batch_size=32, 
            validation_data=(st.session_state.X_test, st.session_state.y_test), 
            verbose=0,  # Disable default logging
            callbacks=[ProgressBar()]  # Progress bar callback
        )

        # Evaluate model
        test_loss, test_mae = model.evaluate(st.session_state.X_test, st.session_state.y_test, verbose=0)

        # Save results to session state
        st.session_state.model = model
        st.session_state.history = history
        st.session_state.test_loss = test_loss
        st.session_state.test_mae = test_mae

        # Display results
        st.write(f"Test MAE: {test_mae:.4f}")
        st.write(f"Test Loss: {test_loss:.4f}")
        st.success("✅ Model training complete!")

#***********************************************************************************************
# predict and plot on 20% unseen data 
# Add Predict and Plot Button
if all(key in st.session_state for key in ['X_test', 'y_test', 'model', 'target_scaler']):
    if st.button("Predict and Plot"):
        # 🚨 GET PREDICTIONS (ON SCALED DATA)
        y_pred_scaled = st.session_state.model.predict(st.session_state.X_test).reshape(-1, 1)

        # 🚨 NOW WE UNSCALE IT PROPERLY
        y_pred = st.session_state.target_scaler.inverse_transform(y_pred_scaled)
        y_actual = st.session_state.target_scaler.inverse_transform(st.session_state.y_test)

        # Create a plot
        plt.figure(figsize=(12, 6))
        plt.plot(y_actual, label=(f"Actual Target Price {ticker}"), color="blue")
        plt.plot(y_pred, label=(f"System Predicted Target Price {ticker}"), linestyle="--", color="red")

        # Add labels and title
        plt.xlabel("Time Step")
        plt.ylabel("Target Price")
        plt.title(f"Actual vs. Predicted Future Price for {ticker}")
        plt.legend()
        plt.grid(True)

        # Show the plot
        st.pyplot(plt)

        st.success("✅ Predictions made and plot displayed.")


