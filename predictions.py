import preprocessing
import pickle
from sklearn.externals import joblib


#load model and label encoder
model = joblib.load('model\\model_config.pkl')

with open('model\\le_conf.pickle','rb') as le_file:
    le = pickle.load(le_file)

def predict(data):
    #preprocessing
    data = preprocessing.define_client_type(data)
    data = data['client'] + ' ' + data['topic'] + ' ' + data['description']
    data = preprocessing.text_preprocessing(data)

    prediction = model.predict([data])
    prediction =  le.inverse_transform(prediction)

    return prediction[0]
    


    