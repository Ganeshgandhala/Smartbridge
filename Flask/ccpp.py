from flask import Flask,render_template,request
import pickle
import numpy as np

app=Flask(__name__) #your application
lr=pickle.load(open('ccpp.pkl','rb'))


@app.route('/') # default route
def home():
    return render_template("ccpp.html")


@app.route('/predict',methods=['post'])
def predict():
    AT=float(request.form['AT'])
    AP=float(request.form['AP'])
    RH=float(request.form['RH'])
    PE=float(request.form['PE'])
    
    
   
    a=np.array([[AT,AP,RH,PE]])
    print(a)
    
    result=lr.predict(a)
    x=result
    return render_template('ccpp.html',x='result is : {}'.format(*x))

if __name__ == '__main__':
    app.run(port=8000) # you are running your app