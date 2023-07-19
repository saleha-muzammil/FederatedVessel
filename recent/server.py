import tensorflow as tf
import tensorflow_federated as tff
from flask import Flask, jsonify, request
import pickle

app = Flask(__name__)

server_state = None

@app.route('/send_server_state', methods=['GET'])
def send_server_state():
    with open("model.pkl", "wb") as f:
        pickle.dump(server_state, f)
    return jsonify({"message": "Server state sent to client successfully."})

@app.route('/receive_client_state', methods=['POST'])
def receive_client_state():
    data = request.get_json()
    client_state = data['state']
    client_metrics = data['metrics']

    # Update the server state with the received client state
    global server_state
    server_state = tff.learning.federated_average(client_state)
    return jsonify({"message": "Client state received and server state updated successfully."})

if __name__ == '__main__':
    # Initialize the global model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(1, input_shape=(55,), kernel_initializer='zeros')
    ])
    server_state = model.get_weights()
    app.run()

