
# from flask import Flask,request, url_for, redirect, render_template
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
app = Flask(__name__)

model = pickle.load(open("CreditCardFraudDetection.pkl","rb"))

@app.route("/")
def CreditCardFraudDetection():
   
    return render_template("index.html")

@app.route("/predict", methods = ["POST"])
def predict():
    loginName = request.form["year"]
    loginName =removeGaps(loginName)
    l = []
    s=""
    for x in loginName:
        if x in[" ",",","	","    "]:
           l.append(float(s))
           s=""
        else:
           s+=x
    l.append(float(s))     

    features = [np.array(l)]
    prediction = model.predict(features)
    if prediction == 0:
        prediction= "Non-Fraudulent Transaction"
    else:
        prediction= "Fraudulent Transaction"
    return render_template("index.html", prediction_text = "Predicted output is :     {}".format(prediction))

def removeGaps(loginName):
    h ="";
    for v in loginName:
        if v in['1','2','3','4','5','6','7','8','9','0','-','.',',']:
            h+=v

    return h
if __name__ == "__main__":
    app.run(host= '0.0.0.0', port= 8080)

