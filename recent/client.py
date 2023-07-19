import tensorflow as tf
import tensorflow_federated as tff
import os
import requests
import pickle

SERVER_URL = os.environ.get('SERVER_URL', 'http://127.0.0.1')
SERVER_PORT = int(os.environ.get('SERVER_PORT', 5000))

def receive_from_server():
    response = requests.get(f"{SERVER_URL}:{SERVER_PORT}/send_server_state")
    with open("model.pkl", "wb") as f:
        f.write(response.content)

def send_this_back_to_server(data):
    requests.post(f"{SERVER_URL}:{SERVER_PORT}/receive_client_state", json=data)

def main():
    # Receive the server_state from the server
    receive_from_server()

    with open("model.pkl", "rb") as f:
        server_state = pickle.load(f)

    # Load the server_state to initialize the client's model
    
    data = pd.read_csv('data_cleaned.csv')

    y_train = data['price']
    X_train = data.drop('price', axis=1)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(1, input_shape=(55,), kernel_initializer='zeros')
    ])
    model.set_weights(server_state)

    # Train the model
    model.fit(X_train_scaled, y_train, epochs=1)

    # Get the client state and metrics
    client_state = model.get_weights()
    client_metrics = model.evaluate(X_train, y_train)  # for simplicity, we use training data for evaluation

    # Serialize the updated 'client_state' and 'client_metrics' to send back to the server
    result = {
        'state': client_state,
        'metrics': client_metrics
    }

    # Send the result back to the server
    send_this_back_to_server(result)

if __name__ == "__main__":
    main()

