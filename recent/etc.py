import collections
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
#from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

# Load Boston Housing Dataset
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

#boston = load_boston()
#data = boston.data
#target = boston.target
train_data, test_data, train_labels, test_labels = train_test_split(data, target, test_size=0.2, random_state=42)

# Normalize data
mean = train_data.mean(axis=0)
std = train_data.std(axis=0)
train_data = (train_data - mean) / std
test_data = (test_data - mean) / std

# Preprocess the data to fit the TFF's input format
train_dataset = tf.data.Dataset.from_tensor_slices((train_data.astype(np.float32), train_labels.astype(np.float32))).batch(len(train_data))
test_dataset = tf.data.Dataset.from_tensor_slices((test_data.astype(np.float32), test_labels.astype(np.float32))).batch(len(test_data))

# Define the collections for client and server data
federated_train_data = [train_dataset]
federated_test_data = [test_dataset]

# Define the model for TFF
def create_tff_model():
    return tff.learning.models.from_keras_model(
        tf.keras.models.Sequential([
            tf.keras.layers.Dense(1, input_shape=(13,))]),
        input_spec=train_dataset.element_spec,
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.MeanSquaredError()])

def client_optimizer_fn():
    return tf.keras.optimizers.SGD(learning_rate=0.01)
def server_optimizer_fn():
    return tf.keras.optimizers.SGD(learning_rate=1.0)

# Define the Federated Averaging process
trainer = tff.learning.algorithms.build_weighted_fed_avg(create_tff_model, client_optimizer_fn=client_optimizer_fn,
    server_optimizer_fn=server_optimizer_fn)

# Train the model on the federated data
state = trainer.initialize()

for _ in range(50):
    state, metrics = trainer.next(state, federated_train_data)
    print('train metrics:', metrics)

# Test the model on the federated data
evaluation_process = tff.learning.algorithms.build_fed_eval(create_tff_model)

#evaluator = tff.learning.build_federated_evaluation(create_tff_model)
evaluation_state = evaluation_process.initialize()


# Extract the model weights from the state
model_weights = trainer.get_model_weights(state)
# Use the model weights in the evaluation
evaluation_state = evaluation_process.set_model_weights(evaluation_state, model_weights)
evaluation_output = evaluation_process.next(evaluation_state, federated_train_data)
str(evaluation_output.metrics)

#test_metrics = evaluator(trained_model_weights, federated_test_data)


#test_metrics = evaluator(state.model, federated_test_data)
print('test metrics:', evaluation_output.metrics)
print('Done')

