import pandas as pd
import preprocessing
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.externals import joblib
from sklearn.linear_model import RidgeClassifier
from matplotlib import pyplot as plt
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.linear_model import SGDClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.neural_network import MLPClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.linear_model import RidgeClassifierCV
# from sklearn.neighbors import NearestCentroid
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.neighbors import RadiusNeighborsClassifier
# from sklearn.ensemble import ExtraTreesClassifier
# from sklearn.semi_supervised import LabelSpreading
# from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


# #load data
# df = pd.read_csv('files\\HD.csv',delimiter=';')

# #preprocessing
# #filter some groups
# df =  df[df.final_unit != '09 Бюджетирование']
# df =  df[df.final_unit != '08 Корректировка данных']
# df =  df[df.final_unit != '07 Видеонаблюдение']
# df =  df[df.final_unit != '04 Снабжения МПР']
# df =  df[df.final_unit != 'Zabbix']

# df = df.apply(preprocessing.define_client_type,axis = 1)
# df['final_unit'] = df['final_unit'].apply(preprocessing.define_work_group)

# df['data'] = df['client'] + ' ' + df['topic'] + ' ' + df['description']
# df = df.dropna()
# df = df.drop(columns=['topic','description','client','contact','service','service_component','unit'],axis = 1)
# df['data'] = df['data'].apply(preprocessing.text_preprocessing)

# with open('files\\prep_HD.pickle','wb') as df_file:
#     pickle.dump(df,df_file)

with open('files\\prep_HD.pickle','rb') as df_file:
    df = pickle.load(df_file)

#distribution plot
df['final_unit'].value_counts().plot(kind = 'bar')
plt.title('distribution - ' + str(len(df)))
plt.show()

df = shuffle(df)

le = LabelEncoder()
Y = le.fit_transform(df['final_unit'])

#split rows for testing
split_len = 1000
data_len = len(Y)
start_slicing = data_len - split_len

X_test = df['data'][start_slicing:]
Y_test = Y[start_slicing:]
doc_numbers_test = df['number'][start_slicing:]

X = df['data'][:start_slicing]
Y = Y[:start_slicing]

# Grid Search for estimators - RidgeClassifier, RandomForestClassifier, MLPClassifier(hidden_layer_sizes=(100,100,100)) are best 
# estimators = (
#     MultinomialNB(),
#     SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42,max_iter=5, tol=None),
#     RandomForestClassifier(),
#     MLPClassifier(hidden_layer_sizes=(100,100,100 )),
#     DecisionTreeClassifier(),
#     RidgeClassifier(),
#     NearestCentroid(),
#     KNeighborsClassifier(),
#     RadiusNeighborsClassifier(),
#     ExtraTreesClassifier(),
#     LabelSpreading(),
#     QuadraticDiscriminantAnalysis())

#model training
estimator =  RidgeClassifier()

text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1,3))), ('tfidf', TfidfTransformer()), ('clf', estimator),])

text_clf.fit(X,Y)

#model evaluation
prediction = text_clf.predict(X_test)
prediction = le.inverse_transform(prediction)
answer =  le.inverse_transform(Y_test)

df_res = pd.DataFrame()
df_res['number'] = doc_numbers_test
df_res['data'] = X_test
df_res['answer'] = answer
df_res['prediction'] = prediction
df_res['accuracy'] = (df_res['answer']  == df_res['prediction'])

df_res = df_res.sort_values(by=['answer']) 

#export res df
writer = pd.ExcelWriter('files\\HD_res.xls')
df_res.to_excel(writer)
writer.save()

accuracy_mes = 'accuracy - {} of {}'.format(df_res['accuracy'].sum(),split_len)

right_df =  df_res.copy()
right_df['right'] = [int(x) for x in df_res['accuracy']]
right_df['wrong'] = [int(x == False) for x in df_res['accuracy']]

right_df = right_df.groupby(['answer']).sum()
right_df = right_df.sort_values(by=['right']) 
right_df.plot(y = ['right','wrong'],kind = 'bar')
plt.title(accuracy_mes)
plt.show()

error_df =  df_res[df_res.accuracy == False]
error_df['error_type'] = error_df['answer'] + ' - ' + error_df['prediction']
error_df['error_type'].value_counts().plot(kind = 'bar')
plt.title('error types')
plt.show()

print(accuracy_mes)
print(metrics.classification_report(answer, prediction,target_names=answer))

#export model config and label encoder
joblib.dump(text_clf, 'model\\model_config.pkl') 
with open('model\\le_conf.pickle','wb') as le_file:
    pickle.dump(le,le_file)


