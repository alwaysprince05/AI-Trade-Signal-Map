import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Load price data (for demo, generate synthetic data)
def load_price_data(n=500):
    np.random.seed(42)
    price = np.cumsum(np.random.randn(n)) + 100
    return pd.DataFrame({'price': price})

# 2. Create features: returns (momentum), rolling volatility
def create_features(df, window=10):
    df['returns'] = df['price'].pct_change().fillna(0)
    df['volatility'] = df['returns'].rolling(window).std().fillna(0)
    return df

# 3. Generate labels: buy/sell/hold (for demo, random logic)
def generate_labels(df):
    # Buy if returns > 0.01, Sell if < -0.01, else Hold
    conds = [df['returns'] > 0.01, df['returns'] < -0.01]
    choices = [2, 0]  # 2=buy, 0=sell, 1=hold
    df['signal'] = np.select(conds, choices, default=1)
    return df

# 4. Train model
MODEL_TYPE = 'RandomForest'  # or 'LogisticRegression'
def train_model(X, y, model_type=MODEL_TYPE):
    if model_type == 'RandomForest':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        model = LogisticRegression(multi_class='multinomial', max_iter=1000)
    model.fit(X, y)
    return model

# 5. Visualization
COLORS = {0: 'blue', 1: 'purple', 2: 'cyan'}
LABELS = {0: 'Sell', 1: 'Hold', 2: 'Buy'}
def plot_signal_map(X, y_pred, model=None, mesh=True):
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 7))
    # Scatter plot
    for label in np.unique(y_pred):
        idx = y_pred == label
        ax.scatter(X[idx, 0], X[idx, 1],
                   c=COLORS[label], label=LABELS[label],
                   alpha=0.6, s=60, edgecolor='none')
    # Optional: smooth decision boundary
    if mesh and model is not None:
        from matplotlib.colors import ListedColormap
        x_min, x_max = X[:, 0].min() - 0.01, X[:, 0].max() + 0.01
        y_min, y_max = X[:, 1].min() - 0.01, X[:, 1].max() + 0.01
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                             np.linspace(y_min, y_max, 200))
        grid = np.c_[xx.ravel(), yy.ravel()]
        zz = model.predict(grid)
        zz = zz.reshape(xx.shape)
        cmap = ListedColormap([COLORS[0], COLORS[1], COLORS[2]])
        ax.contourf(xx, yy, zz, alpha=0.18, cmap=cmap, levels=[-0.5,0.5,1.5,2.5])
    ax.set_xlabel('Momentum (Returns)')
    ax.set_ylabel('Volatility')
    ax.set_title('AI Trade Signal Map')
    ax.legend()
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # Load and process data
    df = load_price_data()
    df = create_features(df)
    df = generate_labels(df)
    # Prepare features and labels
    X = df[['returns', 'volatility']].values
    y = df['signal'].values
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # Train/test split (not strictly needed for visualization)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    # Train model
    model = train_model(X_train, y_train)
    # Predict on all data for visualization
    y_pred = model.predict(X_scaled)
    # Plot
    plot_signal_map(X_scaled, y_pred, model=model, mesh=True)
