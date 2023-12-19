from flask import Flask, jsonify, request, abort
from predict import predict, load_client_data, process_client_data, explain, simulate_predict, analyse_feature

app = Flask(__name__)
app.json.sort_keys = False

@app.route('/clients/<client_id>', methods=['GET'])
def get_client_data(client_id):
    client_data = load_client_data(client_id)

    if client_data.shape[0] == 0:
        abort(404)

    return jsonify(client_data.to_dict())


@app.route('/clients/<client_id>/scores', methods=['GET'])
def get_prediction(client_id):
    scores = predict(client_id)

    return jsonify(scores)

@app.route('/clients/<client_id>/features_explained', methods=['GET'])
def get_features_explained(client_id):
    shap_features = explain(client_id)

    return jsonify(shap_features)


@app.route('/clients/<client_id>/simulate', methods=['POST'])
def simulate_score(client_id):
    payload = request.get_json()

    scores = simulate_predict(payload)

    return jsonify(scores)


@app.route('/features/<feature_name>', methods=['GET'])
def get_feature_profiling(feature_name):
    feature_profiling = analyse_feature(feature_name)

    return jsonify(feature_profiling)


if __name__ == '__main__':
    app.run(debug=True)