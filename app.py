from flask import Flask,request,render_template
import pickle
import numpy as np

app=Flask(__name__)
model=pickle.load(open("model.pkl","rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict",methods=["POST"])
def predict():
    if request.method=="POST":
        preg = int(request.form['Pregnancies'])
        glucose = int(request.form['Glucose'])
        bp = int(request.form['BloodPressure'])
        st = int(request.form['Skinthickness'])
        insulin = int(request.form['Insulin'])
        bmi = float(request.form['BMI'])
        dpf = float(request.form['DPF'])
        age = int(request.form['Age'])
        
        data = np.array([[preg, glucose, bp, st, insulin, bmi, dpf, age]])
        my_prediction = model.predict(data)
        
        return render_template('result.html', prediction=my_prediction)
    
if __name__ == "__main__":
    app.run(debug=True)