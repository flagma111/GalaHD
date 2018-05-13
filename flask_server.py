from flask import Flask,request
import predictions


app = Flask(__name__)

@app.route('/predict',methods=['POST'])
def predict():
    data = request.json
    return predictions.predict(data)

if __name__ == '__main__':
    app.run(host="192.168.51.42")


