import h5py
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, LeakyReLU

# Load the ASCAD dataset
ascad_dataset_path =  r"D:\DSUsers\uig88508\coding_playground\ASCAD\ATMEGA_AES_v1\ATM_AES_v1_fixed_key\ASCAD_data\ASCAD_data\ASCAD_databases\ASCAD.h5"  # Replace with the actual path to your ASCAD dataset
with h5py.File(ascad_dataset_path, 'r') as f:
    profiling_traces = np.array(f['Profiling_traces/traces'], dtype=np.float32)
    profiling_labels = np.array(f['Profiling_traces/metadata']['key'], dtype=np.uint8)
    attack_traces = np.array(f['Attack_traces/traces'], dtype=np.float32)
    attack_labels = np.array(f['Attack_traces/metadata']['key'], dtype=np.uint8)

# Normalize the power traces
def normalize_traces(traces):
    normalized_traces = []
    for trace in traces:
        trace = trace - np.mean(trace)
        min_val = np.min(trace)
        max_val = np.max(trace)
        trace = (trace - min_val) / (max_val - min_val)
        normalized_traces.append(trace)
    return np.array(normalized_traces)

normalized_profiling_traces = normalize_traces(profiling_traces)
normalized_attack_traces = normalize_traces(attack_traces)

# Ensure keys are in the correct format (binary array)
def convert_keys_to_binary(keys):
    binary_keys = []
    for key in keys:
        bin_key = ''.join(format(byte, '08b') for byte in key)
        binary_keys.append(list(map(int, bin_key)))
    return np.array(binary_keys)

profiling_labels = convert_keys_to_binary(profiling_labels)
attack_labels = convert_keys_to_binary(attack_labels)

# Ensure the shape of traces and keys are consistent
assert normalized_profiling_traces.shape[0] == profiling_labels.shape[0], "Number of profiling traces and labels must match"
assert normalized_attack_traces.shape[0] == attack_labels.shape[0], "Number of attack traces and labels must match"

# Prepare data for model training
num_samples, seq_length = normalized_profiling_traces.shape
feature_dim = 1
normalized_profiling_traces = normalized_profiling_traces.reshape((num_samples, seq_length, feature_dim))
normalized_attack_traces = normalized_attack_traces.reshape((normalized_attack_traces.shape[0], seq_length, feature_dim))

# Define and train the model
def train_key_prediction_model(traces, keys):
    model = Sequential()
    model.add(LSTM(256, input_shape=(seq_length, feature_dim)))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dense(keys.shape[1], activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(traces, keys, epochs=10, batch_size=32)
    return model

# Train the prediction model
key_prediction_model = train_key_prediction_model(normalized_profiling_traces, profiling_labels)

# Save the trained model
key_prediction_model.save('key_prediction_model.h5')

print("Model training complete and saved.")

# Use the model to predict keys from new leakage data (attack traces)
predicted_keys = key_prediction_model.predict(normalized_attack_traces)

# Convert predicted binary keys back to integers and validate
def bin_to_int(bin_list):
    bin_list = (bin_list > 0.5).astype(int)  # Ensure binary output
    return int("".join(map(str, bin_list)), 2)

correct_predictions = 0
for pk in predicted_keys:
    predicted_key = [bin_to_int(pk[i*8:(i+1)*8]) for i in range(16)]
    if predicted_key == list(attack_labels[correct_predictions]):
        correct_predictions += 1

print(f"Number of correct predictions: {correct_predictions}/{len(predicted_keys)}")
