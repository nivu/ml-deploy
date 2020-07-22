import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

app = Flask(__name__)

# load the model from disk

# REg
model = pickle.load(open('model.pkl', 'rb'))

# Classification
clf = pickle.load(open('nlp_model.pkl', 'rb'))
cv = pickle.load(open('tranform.pkl', 'rb'))

# iris classification
iris_model = pickle.load(open('models/iris-model.pkl', 'rb'))
iris_target = pickle.load(open('models/target_names.pkl', 'rb'))

# bank XGBoost

bank_xgb = pickle.load(open('models/bank_xg_pipe.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/email')
def email():
    return render_template('home.html')


@app.route('/iris', methods=['POST'])
def iris_predict():
    '''
    For rendering results on HTML GUI
    '''
    format = request.args.get('format')
    print("here", format)
    print("new line")

    # experience = request.form['experience']
    # petal_len = request.form['pl]
    # sl
    # ip_Ary = [pl, sl, pwm sw]

    int_features = [float(x) for x in request.form.values()]
    print(int_features)
    final_features = [np.array(int_features)]
    prediction = iris_model.predict(final_features)

    output = prediction[0]
    target_name = iris_target[output]

    if(format == 'json'):
        return jsonify({'output': int(output), 'target_name': target_name})

    return render_template('index.html', prediction_text='Target class is  {}'.format(output))


@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    print(request.form['test_score'])
    format = request.args.get('format')
    print("here", format)
    print("new line")

    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    if(format == 'json'):
        return jsonify({'salary': output})

    return render_template('index.html', prediction_text='Employee Salary should be $ {}'.format(output))


@app.route('/predict_email', methods=['POST'])
def predict_email():

    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = cv.transform(data).toarray()
        my_prediction = clf.predict(vect)
    return render_template('result.html', prediction=my_prediction)


@app.route('/bank', methods=['POST'])
def predict_bank():

    format = request.args.get('format')

    int_features = [str(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    print(final_features)
    df = pd.DataFrame(final_features, columns=['age', 'job', 'marital', 'education', 'loan', 'duration', 'campaign',
                                               'employee_variation_rate', 'consumer_price_index',
                                               'consumer_confidence_index', 'euribor', 'num_of_employees'])
    prediction = bank_xgb.predict(df)
    print(prediction)

    output = prediction[0]

    if(format == 'json'):
        return jsonify({'bank': int(output)})

    return render_template('index.html', prediction_text='Employee Salary should be $ {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)  # auto-reload on code change
