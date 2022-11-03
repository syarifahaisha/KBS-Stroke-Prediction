from flask import Flask, render_template, request
import pickle
import pandas as pd
from sklearn import preprocessing

app = Flask(__name__)

def load_model():
    classifier_from_pickle = pickle.load(open('../model.pkl', 'rb'))
    print(classifier_from_pickle)
    return classifier_from_pickle

def preprocess_input(input_dict:dict):
    df = pd.read_csv("../dataset/healthcare-dataset-stroke-data.csv")

    #Remove every row that contains NaN value
    df = df.dropna()

    #Remove unecessary column
    df = df.drop('id', axis=1)

    df_input = pd.DataFrame.from_dict([dict], orient='columns')
    
    df = pd.concat([df, df_input])

    le = preprocessing.LabelEncoder()
    df['gender'] = le.fit_transform(df['gender'])
    df['ever_married'] = le.fit_transform(df['ever_married'])
    df['work_type'] = le.fit_transform(df['work_type'])
    df['Residence_type'] = le.fit_transform(df['Residence_type'])
    df['smoking_status'] = le.fit_transform(df['smoking_status'])

    print(df[:10])

    return df.tail(1).values.tolist()

def predict(classifier, input):
    y_pred = classifier.predict(input)

    return y_pred, classifier.predict_proba(input).max()

@app.route('/', methods=['GET', 'POST'])
def index():
    if (request.method == 'POST'):
        input_raw = {}
        input_raw["gender"] = request.values.get('gender')
        input_raw["age"] = request.values.get('age')
        input_raw["hypertension"] = request.values.get('hypertension')
        input_raw["heart_disease"] = request.values.get('heart_disease')
        input_raw["ever_married"] = request.values.get('ever_married')
        input_raw["work_type"] = request.values.get('work_type')
        input_raw["residence_type"] = request.values.get('residence_type')
        input_raw["avg_glukose_level"] = request.values.get('avg_glukose_level')
        input_raw["bmi"] = request.values.get('bmi')
        input_raw["smoking_status"] = request.values.get('smoking_status')

        print(input_raw)

        input = preprocess_input(input_raw)

        model_classifier = load_model()

        predicted_class, confidence = predict(model_classifier, input)

        return render_template("index.html", predicted_class=predicted_class, confidence=confidence)
    else:
        return render_template("index.html", predicted_class=None, confidence=None)