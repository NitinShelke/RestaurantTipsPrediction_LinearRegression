from flask import Flask,request,jsonify,render_template
import numpy as np
import pickle

model=pickle.load(open('model.pkl','rb'))

app = Flask(__name__)


@app.route('/')
def hello_world():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    initial_features=[int(x) for x in request.form.values()]
    final_features=[np.array(initial_features)]
    prediction=model.predict(final_features)

    return  render_template('index.html',prediction_text='tip would be {}'.format(prediction))



if __name__ == '__main__':
    app.run(debug=True)