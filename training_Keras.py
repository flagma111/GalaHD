import pandas as pd
import re
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding
from keras.layers import LSTM, SpatialDropout1D
from keras.utils import to_categorical
from matplotlib import pyplot as plt
import pickle

def text_preprocessing(x):
    x = re.sub(r'[;,?!."><-_:№%()]',' ', x).strip()
    x = x.lower()
    x = x.replace('1 С','одинэс')
    x = x.replace('1С','одинэс')
    x = x.replace('1с','одинэс')
    x = x.replace('1c','одинэс')
    x = x.replace('1с8','одинэс')
    x = x.replace('1c8','одинэс')
    x = x.replace('\n',' ')
    x = x.replace(' для ',' ')
    x = x.replace(' не ',' ')
    x = x.replace(' за ',' ')
    x = x.replace(' нет ',' ')
    x = x.replace(' на ',' ')
    x = x.replace(' в ',' ')
    x = x.replace(' о ',' ')
    x = x.replace(' с ',' ')
    x = x.replace(' а ',' ')
    x = x.replace(' но ',' ')
    x = x.replace(' и ',' ')
    x = x.replace(' или ',' ')
    x = x.replace(' у ',' ')
    x = x.replace(' к ',' ')
    x = x.replace(' т ',' ')
    x = x.replace(' г ',' ')
    x = x.replace(' по ',' ')
    x = x.replace(' от ',' ')
    x = x.replace(' из ',' ')
    x = x.replace(' так ',' ')
    x = x.replace(' как ',' ')
    x = x.replace(' он ',' ')
    x = re.sub(r'\d+','', x)
    x = re.sub('(\\b[A-Za-z] \\b|\\b [A-Za-z]\\b)', '', x)

    signature = x.find(' уважением')

    if signature != -1:
        x = x[:signature-1]
    
    signature = x.find(' from:')

    if signature != -1:
        x = x[:signature-1]

    signature = x.find(' mailto')

    if signature != -1:
        x = x[:signature-1]  

    return x 

def define_client_type(x):
    if x.startswith(('ЕКБ','МСК','НСК','Минск')):
        return 'ЭтоМПР'
    else:
        return 'ЭтоНеМПР'

def define_work_group(x):
    if x.startswith('РЦ'):
        return 'РЦ'
    else:
        return x
    
df = pd.read_csv('files\\HD.csv',delimiter=';')

#preprocessing

#filter some groups
df =  df[df.ОргПодразделение != '09 Бюджетирование']
df =  df[df.ОргПодразделение != '07 Видеонаблюдение']

df = shuffle(df)
df['Клиент'] = df['Клиент'].apply(define_client_type)
df['ФинальнаяРабочаяГруппа'] = df['ФинальнаяРабочаяГруппа'].apply(define_work_group)

#distribution plot
# distribution_df = df.copy()
# distribution_df['res'] = 1
# distribution_df = distribution_df.groupby(['ФинальнаяРабочаяГруппа']).sum()
# distribution_df = distribution_df.sort_values(by=['res']) 
# distribution_df.plot(y = 'res',kind = 'bar')
# plt.title('distribution')
# plt.show()

df['data'] = df['Клиент'] + ' ' + df['ТемаОбращения'] + ' ' + df['Описание']
# df['data'] =  df['ТемаОбращения'] + ' ' + df['Описание']

df = df.dropna()
df = df.drop(columns=['ТемаОбращения', 'Описание','Клиент','КонтактноеЛицо','КомпонентаУслуги','Услуга','ОргПодразделение'],axis = 1)
df['data'] = df['data'].apply(text_preprocessing)

df.to_csv('files\\prep_HD.csv')

num_words = 2200
maxlen = 20

# while num_words <= 3000:
#     print(num_words,maxlen)

tokenizer = Tokenizer(char_level=False,num_words = num_words,split=' ')
tokenizer.fit_on_texts(df['data'])

df['token_data'] = tokenizer.texts_to_sequences(df['data'])
X = pad_sequences(df['token_data'], maxlen=maxlen)

le = LabelEncoder()
Y = to_categorical(le.fit_transform(df['ФинальнаяРабочаяГруппа']))

#split rows for testing
split_len = 500
data_len = len(Y)
start_slicing = data_len - split_len

X_test = X[start_slicing:]
Y_test = Y[start_slicing:]

X = X[:start_slicing]
Y = Y[:start_slicing]

inp_count = len(Y[0])

#training
model = Sequential()
model.add(Embedding(num_words, 10))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(200, dropout=0.2, recurrent_dropout=0.2)) 
model.add(Dense(inp_count, activation="softmax"))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(X, Y, batch_size=64, epochs=8, validation_data=(X_test,Y_test), verbose=2)

    # num_words = num_words + 200

# prediction
prediction = model.predict(X_test,batch_size=64)
prediction = [np.argmax(x) for x in prediction]
prediction = le.inverse_transform(prediction)
answer = [np.argmax(x) for x in Y_test]
answer =  le.inverse_transform(answer)

df = pd.DataFrame()
df['answer'] = answer
df['prediction'] = prediction
df['accuracy'] = (df['answer']  == df['prediction'])

accuracy_mes = 'accuracy - {} of {}'.format(df['accuracy'].sum(),split_len)

error_df =  df[df.accuracy == False]
error_df['error_type'] = error_df['answer'] + ' - ' + error_df['prediction']
error_df['res'] = 1

error_df = error_df.groupby(['error_type']).sum()
error_df = error_df.sort_values(by=['res']) 
error_df.plot(y = 'res',kind = 'bar')
plt.title(accuracy_mes)
plt.show()

print(accuracy_mes)

# export
df.to_csv('files\\HD_res.csv')

# with open('model\\le_conf.pickle','wb') as le_file:
#     pickle.dump(le,le_file)

# with open('model\\tokenizer_conf.pickle','wb') as tokenizer_file:
#     pickle.dump(tokenizer,tokenizer_file)

# model_json = model.to_json()
# json_file = open('model\\model_configuration.json',"w")
# json_file.write(model_json)
# json_file.close()
        
# model.save_weights('model\\weights.h5') 
