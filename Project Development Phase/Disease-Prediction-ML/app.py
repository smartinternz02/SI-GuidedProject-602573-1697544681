from flask import Flask, render_template, request
import numpy as np
import pickle

model = pickle.load(open('model.pkl','rb'))
app = Flask(__name__)

@app.route("/")
def home():
    return render_template('index.html')

@app.route('/details')
def pred():
    return render_template('details.html')

@app.route('/predict',methods=['POST','GET'])
def predict():
    col=['chills', 'joint_pain', 'vomiting', 'fatigue', 'weight_loss',
       'restlessness', 'lethargy', 'cough', 'high_fever', 'sweating',
       'headache', 'dark_urine', 'nausea', 'loss_of_appetite',
       'pain_behind_the_eyes', 'back_pain', 'diarrhoea', 'mild_fever',
       'yellowing_of_eyes', 'blurred_and_distorted_vision', 'phlegm',
       'congestion', 'chest_pain', 'fast_heart_rate', 'puffy_face_and_eyes',
       'excessive_hunger', 'knee_pain', 'muscle_weakness', 'stiff_neck',
       'swelling_joints', 'loss_of_balance', 'unsteadiness',
       'bladder_discomfort', 'passage_of_gases', 'depression', 'irritability',
       'muscle_pain', 'abnormal_menstruation', 'increased_appetite',
       'family_history', 'mucoid_sputum', 'rusty_sputum',
       'lack_of_concentration', 'receiving_blood_transfusion', 'coma',
       'history_of_alcohol_consumption', 'blood_in_sputum', 'palpitations',
       'inflammatory_nails']
    if request.method=='POST':
        inputt = [str(x) for x in request.form.values()]

        b=[0]*49
        for x in range(0,49):
            for y in inputt:
                if(col[x]==y):
                    b[x]=1
        b=np.array(b)
        b=b.reshape(1,49)
        prediction = model.predict(b)
        prediction = prediction[0]
    return render_template('results.html', prediction_text="The probable diagnosis says it could be {}".format(prediction))

if __name__ == "__main__":
    app.run()

    