import re
import pymorphy2


morph = pymorphy2.MorphAnalyzer()

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

    x = ' '.join([morph.parse(z)[0].normal_form for z in x.split()])

    return x 

def define_client_type(row):
    if row['client'].startswith(('ЕКБ','МСК','НСК','Минск')) or row['contact'].startswith('ФР'):
        row['client'] =  'ЭтоМПР'
    elif row['client'] == '01-Индустриальный1' or row['client'] == '04-Томилино43' or row['client'] == '01-Петухова71':
        row['client'] = 'ЭтоСКЛАД'
    else:
        row['client'] = 'ЭтоОФИС'
    
    return row

def define_work_group(x):
    if x.startswith('РЦ'):
        return 'РЦ'
    else:
        return x
    