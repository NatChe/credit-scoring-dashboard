from flask import Flask, jsonify, request
from predict import predict

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def get_prediction():
    client_id = request.form.get('client_id')
    scores = predict(client_id)

    return jsonify(scores)


if __name__ == '__main__':
    app.run(debug=True)