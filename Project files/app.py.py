from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the saved model (using the full path you've been using)
model_path = r'C:\Users\surya\Desktop\Wind_Power_Project\power_prediction.sav'
model = pickle.load(open(model_path, 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Capture the three inputs from the form
    try:
        features = [float(x) for x in request.form.values()]
        final_features = [np.array(features)]
        
        # Predict output
        prediction = model.predict(final_features)
        output = round(prediction[0], 2)

        return render_template('index.html', 
                               prediction_text=f'Predicted Energy Output: {output} kWh')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == "__main__":
    # use_reloader=False is required to run Flask inside Spyder properly
    app.run(debug=True, use_reloader=False)