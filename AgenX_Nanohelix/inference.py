import numpy as np
import pandas as pd
import joblib
import optuna
import warnings
import os
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score

warnings.filterwarnings("ignore")


# Function to compute derived parameters from the 4 input parameters
def compute_nanohelix_parameters(df):
    # Create a copy to avoid modifying the original
    df_enriched = df.copy()

    # Calculate derived parameters using vectorized operations
    pitch = df_enriched['pitch']
    fiber_radius = df_enriched['fiber_radius']
    n_turns = df_enriched['n_turns']
    helix_radius = df_enriched['helix_radius']

    # Calculate turn length for all rows at once
    turn_length = np.sqrt((2 * np.pi * helix_radius) ** 2 + pitch ** 2)

    # Add all derived parameters using vectorized operations
    df_enriched['total_length'] = turn_length * n_turns
    df_enriched['height'] = pitch * n_turns
    df_enriched['curl'] = helix_radius / (helix_radius ** 2 + (pitch / (2 * np.pi)) ** 2)
    df_enriched['angle'] = np.arctan2(pitch, 2 * np.pi * helix_radius)
    df_enriched['total_fiber_length'] = df_enriched['total_length'] * (1 + (2 * np.pi * fiber_radius) / turn_length)
    df_enriched['V'] = np.pi * fiber_radius ** 2 * df_enriched['total_fiber_length']
    df_enriched['mass'] = df_enriched['V']

    return df_enriched

# Load and prepare the data
def load_and_prepare_data():
    """
    Load and prepare the data for training.

    Returns:
    --------
    X_train, X_test, y_train, y_test, scaler_X, scaler_y
    """
    # Load data
    try:
        train_data = pd.read_csv('data/train_g_0603.csv')
        test_data = pd.read_csv('data/test_g_0603.csv')

        print("Dataset loaded successfully.")
    except FileNotFoundError:
        print("Warning: Dataset files not found. Creating dummy data for demonstration.")

        # Create dummy data for demonstration
        np.random.seed(42)
        train_size, test_size = 800, 200

        # Generate random parameters
        train_data = pd.DataFrame({
            'pitch': np.random.uniform(50, 300, train_size),
            'fiber_radius': np.random.uniform(10, 50, train_size),
            'n_turns': np.random.uniform(1, 5, train_size),
            'helix_radius': np.random.uniform(50, 150, train_size),
        })

        # Add dummy g_factor (target)
        train_data['g_factor'] = (
                0.5 * train_data['pitch'] +
                0.3 * train_data['fiber_radius'] +
                0.2 * train_data['n_turns'] +
                0.4 * train_data['helix_radius'] +
                np.random.normal(0, 0.5, train_size)
        )

        # Generate test data similarly
        test_data = pd.DataFrame({
            'pitch': np.random.uniform(50, 300, test_size),
            'fiber_radius': np.random.uniform(10, 50, test_size),
            'n_turns': np.random.uniform(1, 5, test_size),
            'helix_radius': np.random.uniform(50, 150, test_size),
        })

        test_data['g_factor'] = (
                0.5 * test_data['pitch'] +
                0.3 * test_data['fiber_radius'] +
                0.2 * test_data['n_turns'] +
                0.4 * test_data['helix_radius'] +
                np.random.normal(0, 0.5, test_size)
        )

    # Extract basic parameters
    basic_params = ['pitch', 'fiber_radius', 'n_turns', 'helix_radius']

    # Compute derived parameters
    print("Computing derived parameters...")
    train_data_enriched = compute_nanohelix_parameters(train_data[basic_params])
    test_data_enriched = compute_nanohelix_parameters(test_data[basic_params])

    # Add any additional parameters from the original dataset that weren't derived
    if 'direction' in train_data.columns and 'direction' not in train_data_enriched.columns:
        train_data_enriched['direction'] = train_data['direction']
        test_data_enriched['direction'] = test_data['direction']

    if 'x_y' in train_data.columns and 'x_y' not in train_data_enriched.columns:
        train_data_enriched['x_y'] = train_data['x_y']
        test_data_enriched['x_y'] = test_data['x_y']

    # Prepare feature matrix and target vector
    X_train = train_data_enriched
    y_train = train_data['g_factor'].values

    X_test = test_data_enriched
    y_test = test_data['g_factor'].values

    # Scale the features and target
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

    print(f"Training set shape: {X_train_scaled.shape}")
    print(f"Testing set shape: {X_test_scaled.shape}")

    return (X_train, X_test, y_train, y_test,
            X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled,
            scaler_X, scaler_y)


# Define Optuna objective for MLP
def objective_mlp(trial):
    # Define hyperparameter search space
    hidden_layer_sizes = tuple(
        trial.suggest_int(f"n_units_l{i}", 32, 512)
        for i in range(trial.suggest_int("n_layers", 1, 4))
    )

    learning_rate_init = trial.suggest_float("learning_rate_init", 1e-5, 1e-2, log=True)
    alpha = trial.suggest_float("alpha", 1e-6, 1e-2, log=True)
    activation = trial.suggest_categorical("activation", ["relu", "tanh"])
    solver = trial.suggest_categorical("solver", ["adam", "sgd"])
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])

    # Create and train model
    model = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        solver=solver,
        alpha=alpha,
        batch_size=batch_size,
        learning_rate_init=learning_rate_init,
        max_iter=500,
        early_stopping=True,
        random_state=42
    )

    # Perform cross-validation
    cv_scores = cross_val_score(
        model, X_train_scaled, y_train_scaled,
        cv=5, scoring="r2", n_jobs=-1
    )

    # Return mean CV score
    return np.mean(cv_scores)


# Train and evaluate the model
def train_and_evaluate():
    """
    Train and evaluate MLP model using Optuna for hyperparameter optimization.
    """
    print("Starting hyperparameter optimization...")

    # Create a directory for models if it doesn't exist
    if not os.path.exists("models"):
        os.makedirs("models")

    # Optimize MLP model
    print("\nOptimizing MLP model...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective_mlp, n_trials=10)
    print(f"Best MLP parameters: {study.best_params}")

    # Create and train best MLP
    best_mlp = MLPRegressor(
        hidden_layer_sizes=tuple(
            study.best_params[f"n_units_l{i}"]
            for i in range(study.best_params["n_layers"])
        ),
        activation=study.best_params["activation"],
        solver=study.best_params["solver"],
        alpha=study.best_params["alpha"],
        batch_size=study.best_params["batch_size"],
        learning_rate_init=study.best_params["learning_rate_init"],
        max_iter=1000,
        random_state=42
    )

    # Train the model on the full training set
    print("\nTraining final model with best parameters...")
    best_mlp.fit(X_train_scaled, y_train_scaled)

    # Evaluate on test set
    y_pred_scaled = best_mlp.predict(X_test_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Test MSE: {mse:.6f}")
    print(f"Test R²: {r2:.6f}")

    # Save model and scalers
    print("\nSaving model and scalers...")
    joblib.dump(best_mlp, "models/nanohelix_mlp_model.pkl")
    joblib.dump(scaler_X, "models/nanohelix_scaler_X.pkl")
    joblib.dump(scaler_y, "models/nanohelix_scaler_y.pkl")
    print("Model and scalers saved successfully.")

    return best_mlp, scaler_X, scaler_y


# Function to predict g-factor
def predict_g_factor(model, scaler_X, scaler_y, pitch, fiber_radius, n_turns, helix_radius, x_y=None, direction=None):
    """
    Predict g-factor using the trained model.

    Parameters:
    -----------
    model : trained model
        The trained MLP model
    scaler_X, scaler_y : StandardScaler
        Scalers for features and target
    pitch, fiber_radius, n_turns, helix_radius : float
        Basic parameters of the nanohelix
    x_y, direction : float, optional
        Additional parameters if used in the original model

    Returns:
    --------
    g_factor : float
        Predicted g-factor
    all_params : dict
        Dictionary with all computed parameters
    """
    # Create a DataFrame with the basic parameters
    data = pd.DataFrame({
        'pitch': [pitch],
        'fiber_radius': [fiber_radius],
        'n_turns': [n_turns],
        'helix_radius': [helix_radius]
    })

    # Add additional parameters if provided
    if x_y is not None:
        data['x_y'] = [x_y]

    if direction is not None:
        data['direction'] = [direction]

    # Compute derived parameters
    data_enriched = compute_nanohelix_parameters(data)

    # #### Do not change the order!#####
    data_enriched['direction'] = 1
    data_enriched['x_y'] = 0

    print(data_enriched.keys())

    # Store all parameters for return
    all_params = data_enriched.iloc[0].to_dict()

    # Scale features
    X_scaled = scaler_X.transform(data_enriched)

    # Make prediction
    y_pred_scaled = model.predict(X_scaled)

    # Inverse transform prediction
    g_factor = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))[0][0]

    # Add g-factor to parameters
    all_params['g_factor'] = g_factor

    return g_factor, all_params


def visualize(y_test, y_pred):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    from sklearn.metrics import r2_score

    # Set the seaborn style
    sns.set(style="whitegrid")

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))

    # Calculate R2 score
    r2 = r2_score(y_test, y_pred)

    # Create scatter plot of predicted vs actual values
    scatter = ax.scatter(y_test, y_pred, alpha=0.6,
                         cmap='cividis', c=np.abs(y_test - y_pred))

    # Add perfect prediction line
    min_val = min(min(y_test), min(y_pred))
    max_val = max(max(y_test), max(y_pred))
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2)

    # Add colorbar to show prediction error magnitude
    cbar = plt.colorbar(scatter)
    cbar.set_label('Absolute Error')

    # Add the R² value to the plot
    ax.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax.transAxes,
            fontsize=14, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    # Labels and title
    ax.set_xlabel('Actual g-factor', fontsize=12)
    ax.set_ylabel('Predicted g-factor', fontsize=12)
    ax.set_title('Model Performance: Predicted vs Actual g-factor', fontsize=14)

    # Set equal aspect ratio
    ax.set_aspect('equal', 'box')

    # Save the figure (using the preferences)
    plt.savefig('r2_plot.pdf', dpi=300, format='pdf', bbox_inches='tight')

    # Show the plot
    plt.show()


# Main function
def main():
    print("=== Nanohelix G-Factor Prediction Model ===")

    # Load and prepare data
    global X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, scaler_X, scaler_y
    (X_train, X_test, y_train, y_test,
     X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled,
     scaler_X, scaler_y) = load_and_prepare_data()

    # Check if model is already trained
    if (os.path.exists("models/nanohelix_mlp_model.pkl") and
            os.path.exists("models/nanohelix_scaler_X.pkl") and
            os.path.exists("models/nanohelix_scaler_y.pkl")):

        print("\nPre-trained model found. Loading model...")

        # Load model and scalers
        model = joblib.load("models/nanohelix_mlp_model.pkl")
        scaler_X = joblib.load("models/nanohelix_scaler_X.pkl")
        scaler_y = joblib.load("models/nanohelix_scaler_y.pkl")

        # Evaluate model
        y_pred_scaled = model.predict(X_test_scaled)
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        r2 = r2_score(y_test, y_pred)

        visualize(y_test, y_pred)

        print(f"Loaded model test R²: {r2:.6f}")


if __name__ == "__main__":
    main()