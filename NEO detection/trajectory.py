import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.regularizers import l2
data=pd.read_csv("neo_data.csv")
data.head()
data= data.drop(['Min Diameter (feet)','Max Diameter (feet)', 'Min Diameter (m)','Max Diameter (m)','Min Diameter (miles)','Max Diameter (miles)', 'Relative Velocity (miles/h)','Miss Distance (astronomical)','Miss Distance (lunar)'], axis=1)
data= data.drop(['ID', 'Neo Reference ID', 'Name', 'Limited Name', 'NASA JPL URL','Close Approach Date'], axis=1)
data['Is Potentially Hazardous'] = data['Is Potentially Hazardous'].astype(int)
data['Orbiting Body'] = data['Orbiting Body'].astype('category').cat.codes
data['Close Approach Date (Full)'] = pd.to_datetime(data['Close Approach Date (Full)']).astype(np.int64) // 10**9
columns_to_normalize= ['Min Diameter (km)','Max Diameter (km)','Relative Velocity (km/s)','Relative Velocity (km/h)','Miss Distance (km)','Miss Distance (miles)' ]
scaler = StandardScaler()
data[columns_to_normalize] = scaler.fit_transform(data[columns_to_normalize])
print(data.head())

data = data.sort_values(by='Epoch Date Close Approach')
values = data[['Relative Velocity (km/s)', 'Miss Distance (km)',
               'Min Diameter (km)', 'Max Diameter (km)']].values

sequence_length = 5  # Number of timesteps to look back
X_sequences, y_sequences = [], []

for i in range(len(values) - sequence_length):
    X_sequences.append(values[i:i+sequence_length])  # Past sequence
    y_sequences.append(values[i+sequence_length, :3])  # Next timestep's 'Miss Distance (km)'

X_sequences = np.array(X_sequences)
y_sequences = np.array(y_sequences)

# Reshape X for LSTM: (samples, timesteps, features)
print("Shape of X:", X_sequences.shape)  # (num_samples, sequence_length, num_features)
print("Shape of y:", y_sequences.shape)
X_reshaped = X_sequences.reshape(-1, X_sequences.shape[2])  # Flatten to 2D for scaling
X_normalized = scaler.fit_transform(X_reshaped)
X_sequences = X_normalized.reshape(X_sequences.shape)
X_train, X_test, y_train, y_test = train_test_split(
    X_sequences, y_sequences, test_size=0.2, random_state=42
)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

model = Sequential([
    LSTM(128, activation='relu', input_shape=(sequence_length, X_sequences.shape[2]), return_sequences=True),
    Dropout(0.3),
    BatchNormalization(),
    LSTM(64, activation='relu',  return_sequences=True), #Added return_sequences=True to connect to the next LSTM layer
    LSTM(64, activation='relu', kernel_regularizer=l2(1e-4)),
    Dropout(0.3),
    Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae','mse'])
model.summary()
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    verbose=1
)

loss = model.evaluate(X_test, y_test)
print("Test Loss:", loss)

y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2= r2_score(y_test, y_pred)
print("Root Mean Squared Error (RMSE):", rmse)
print("R-squared (R2) Score:", r2)

plt.plot(y_test, label='Actual Trajectory')
plt.plot(y_pred, label='Predicted Trajectory')
plt.legend()
plt.show()

model.save('trajectory_model.keras')