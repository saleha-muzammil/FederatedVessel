# previous code included
import tensorflow as tf
import tensorflow_federated as tff
import pickle 
import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
import json
import joblib
from flask import Flask, jsonify, request

app = Flask(__name__)

server_state= None 

def send_to_client(data):
    print ('sending to client')
    with open("model.pkl", "wb") as f:
        pickle.dump(data, f)

def get_result_from_client():
    print('receiving from client') 
    with open("client_result.pkl", "rb") as f:
        return pickle.load(f)

# Endpoint to send the 'server_state' to the client
@app.route('/send_server_state', methods=['GET'])
def send_server_state():
    send_to_client(server_state)
    return jsonify({"message": "Server state sent to client successfully."})

# Endpoint to receive the client result and update the 'server_state'
@app.route('/receive_client_result', methods=['POST'])
def receive_client_result():
    data = request.get_json()
    client_state = data['state']
    client_metrics = data['metrics']

    # Update the server state with the received client state
    global server_state
    server_state = client_state

    return jsonify({"message": "Client result received and server state updated successfully."})

if __name__ == '__main__':
    data= pd.read_csv('data_cleaned.csv')
    label = data['price']
    features = data.drop('price', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = Ridge(alpha=10)
    model.fit(X_train, y_train)
    #server_state = model.get_weights()
    joblib.dump(model, 'model.pkl')
    send_to_client(model)
    get_result_from_client()
    app.run()
