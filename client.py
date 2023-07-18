import tensorflow as tf
import tensorflow_federated as tff
import pandas as pd
import os
import requests
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

SERVER_URL = os.environ.get('SERVER_URL', 'http://127.0.0.1')
SERVER_PORT = int(os.environ.get('SERVER_PORT', 5000))

def receive_from_server():
    response = requests.get(f"{SERVER_URL}:{SERVER_PORT}/send_server_state")
    with open("model.pkl", "wb") as f:
        f.write(response.content)

def send_this_back_to_server(data):
    requests.post(f"{SERVER_URL}:{SERVER_PORT}/receive_client_result", json=data)

def main():
    data = pd.read_csv('data_cleaned.csv')

    label = data['price']
    features = data.drop('price', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Receive the server_state from the server
    receive_from_server()

    # Load the server_state to initialize the client's model

    # Define the TFF types for model and dataset

    model = Ridge(alpha=10)
# Train the model
    model.fit(X_train, y_train)

    # Create the TFF dataset
    tff_train_dataset = tff.tf_computation(lambda: tf.data.Dataset.from_tensor_slices((X_train_scaled, y_train)).batch(32))()

    # Define the TFF iterative process
    #iterative_process = tff.learning.algorithms.build_weighted_fed_avg(model_fn=lambda: tff_model_type, client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.1))

    # Initialize the client state with the server state
#    state = iterative_process.initialize()
#    state = tff.learning.state_with_new_model_weights(state, server_state)

    # Run federated learning rounds
   # NUM_ROUNDS = 10
   # for round_num in range(NUM_ROUNDS):
    #    state, metrics = iterative_process.next(state, [tff_train_dataset])

    # Get the client state and metrics
  #  client_state = tff.learning.state_with_new_model_weights(state.model, state.optimizer_state)
   # client_metrics = metrics

    # Serialize the updated 'client_state' and 'client_metrics' to send back to the server
    result = {
        'state': 1 , #client_state,
        'metrics': 2 #client_metrics
    }

    # Send the result back to the server
    send_this_back_to_server(result)
    print('Sending client_state to the server.')

if __name__ == "__main__":
    print('HELLO')
    print(SERVER_URL, SERVER_PORT)
    main()

