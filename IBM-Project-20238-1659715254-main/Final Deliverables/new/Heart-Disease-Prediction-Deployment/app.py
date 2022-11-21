# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the Random Forest CLassifier model
filename = 'heart-disease-prediction-knn-model.pkl'
model = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/')
@app.route('/home')
def home():
	return render_template('index.html')

@app.route('/login')
def login():
	return render_template('login.html')

@app.route('/main')
def main():
    return render_template('main.html')

@app.route('/exercise')
def exercise():
    return render_template('exercise.html')

@app.route('/foods')
def foods():
    return render_template('foods.html')

@app.route('/signup')
def signup():
    return render_template('signup.html')


@app.route('/dash')
def dash():
    return render_template('dash.html')
@app.route('/facts')
def facts():
    return render_template('facts.html')
@app.route('/bluebc')
def bluebc():
	return render_template('../imaggee/bluebc.jpg')

@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':

        age = int(request.form['age'])
        sex = request.form.get('sex')
        cp = request.form.get('cp')
        trestbps = int(request.form['trestbps'])
        chol = int(request.form['chol'])
        fbs = request.form.get('fbs')
        restecg = int(request.form['restecg'])
        thalach = int(request.form['thalach'])
        exang = request.form.get('exang')
        oldpeak = float(request.form['oldpeak'])
        slope = request.form.get('slope')
        ca = int(request.form['ca'])
        thal = request.form.get('thal')
        
        data = np.array([[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]])
        my_prediction = model.predict(data)
        
        return render_template('result.html', prediction=my_prediction)
        
        

if __name__ == '__main__':
	app.run(debug=True)

