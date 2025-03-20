import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from stable_baselines3 import PPO
import gym

# LSTM Model for Demand Forecasting
def train_lstm_model():
    # Prepare data for LSTM
    def create_dataset(data, look_back=30):
        X, y = [], []
        for i in range(len(data) - look_back):
            X.append(data[i:(i + look_back)])
            y.append(data[i + look_back])
        return np.array(X), np.array(y)

    # Load processed data
    data = pd.read_csv("Datasets/processed_blood_data.csv")
    data = data['demand'].values.reshape(-1, 1)

    # Create training and testing datasets
    look_back = 30
    X, y = create_dataset(data, look_back)
    X_train, X_test = X[:int(0.8 * len(X))], X[int(0.8 * len(X)):]
    y_train, y_test = y[:int(0.8 * len(y))], y[int(0.8 * len(y)):]

    # Build LSTM model
    model = Sequential()
    model.add(LSTM(50, input_shape=(look_back, 1)))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')

    # Train the model
    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

    # Save the model
    model.save("Trained_Model/blood_demand_lstm.h5")

# RL Model for Inventory Optimization
def train_rl_model():
    # Define custom blood bank environment
    class BloodBankEnv(gym.Env):
        def __init__(self):
            super(BloodBankEnv, self).__init__()
            self.action_space = gym.spaces.Discrete(3)  # Actions: order more, maintain, reduce
            self.observation_space = gym.spaces.Box(low=0, high=100, shape=(1,))

        def step(self, action):
            # Implement logic for inventory management
            pass

        def reset(self):
            # Reset environment state
            pass

    # Create and train RL model
    env = BloodBankEnv()
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000)

    # Save the model
    model.save("Trained_Model/inventory_optimization_rl")

# Train both models
train_lstm_model()
train_rl_model()