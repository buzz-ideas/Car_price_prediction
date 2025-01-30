from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model (make sure the file name is correct)
with open("Car.predict.pkl", "rb") as file:
    grade_model = pickle.load(file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/aboutus')
def aboutus():
    return render_template('aboutus.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    prediction = None
    if request.method == 'POST':
        # Extract the values from the form
        seller_type = int(request.form['seller_type'])
        owner = int(request.form['owner'])
        transmission = int(request.form['transmission'])
        fuel = int(request.form['fuel'])
        year = int(request.form['year'])
        kilometers_driven = int(request.form['kilometers_driven'])

        # Prepare the input data in the same format as the training data
        input_data = np.array([[seller_type, owner, transmission, fuel, year, kilometers_driven]])

        # Make a prediction using the loaded model
        prediction = grade_model.predict(input_data)[0]

    return render_template('predict.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
