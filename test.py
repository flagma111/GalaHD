import requests
import json


url_srv = 'http://127.0.0.1:5000'
url = url_srv + '/predict'


client = 'ЕКБ Белореченская'
topic = 'термопринтер'
contact = ''
description = '''ЕКБ Белореченская_термопринтер зала 
    Прошу заменить  термопринтер в зале( Xprinter ) , т.к. он не работает 

    С Уважением, 
    Белопашенцева Юлия 
    Управляющий ЕКБ Белореченская 
    Сеть ""ГАЛАМАРТ"" 
    Магазин Постоянных Распродаж 
    8 (343) 378-40-03 доб.4100 
    8 (912) 249 69 06 
    direktor-090@galamart.ru'''

# data = json.dumps({'client':client,'topic':topic,'description':description})
# res = requests.get(url = url, data = data.encode('utf-8'))

res = requests.post(url = url, json = {'client':client,'contact':contact,'topic':topic,'description':description})
 
print(res.text)

client = 'ЕКБ Ижевск Автозаводская'
topic = 'не заряжает ТСД'
contact = ''
description = '''ТСД перестал заряжатся. 
Его нужно отправлять Вам либо можно отремонтировать самим? 

С уважением, ЗУМ «Галамарт» 
Клевцова Елизавета Николаевна. 
Тел.(3412)93-62-13 
Т.89090525751 
Т.89821251569'''

# data = json.dumps({'client':client,'topic':topic,'description':description})
# res = requests.get(url = url, data = data.encode('utf-8'))

res = requests.post(url = url, json = {'client':client,'contact':contact,'topic':topic,'description':description})
 
print(res.text)


client = 'Подрядчики'
topic = 'galacentre.ru | Синхронизация'
contact = ''
description = '''27.04.2018 17:56:01:  Ошибка:  Ошибка распаковки файла. Файл выгрузки отсутствует или битый! 

Сообщение сгенерировано автоматически.'''

# data = json.dumps({'client':client,'topic':topic,'description':description})
# res = requests.get(url = url, data = data.encode('utf-8'))

res = requests.post(url = url, json = {'client':client,'contact':contact,'topic':topic,'description':description})

print(res.text)


client = 'Пользователи'
topic = 'добавление нового ИП в узел Излишки'
contact = 'Бовина Алена'
description = '''В связи регистрацией нового ИП и началом работы по направлению Росзакуп Без НДС прошу добавить в узел Излишки бух-ию нового ИП на УСН  - ИП Пирогов Н.Е. 


С уважением, 
Главный бухгалтер Бовина Алена 
Тел. (343) 378-40-03 (доб. 4810) 
e-mail: alena.bovina@mympr.ru'''

# data = json.dumps({'client':client,'topic':topic,'description':description})
# res = requests.get(url = url, data = data.encode('utf-8'))

res = requests.post(url = url, json = {'client':client,'contact':contact,'topic':topic,'description':description})

print(res.text)

