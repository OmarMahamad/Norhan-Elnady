from flask import Flask, request, jsonify
import joblib

model = joblib.load("model.pkl")

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    prediction = model.predict([data["features"]])
    return jsonify(prediction.tolist())

@app.route("/",methods=["GET"])
def mayname():
    return "My name is nooor"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)