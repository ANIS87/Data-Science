import unicodedata
import bson
from ast import literal_eval
import json, ast 
import collections
import StringIO, csv
import datetime, time
from bsonstream import KeyValueBSONInput
#with open('shop_match.bson','rb') as f:
    #data = bson.decode_all(f.read())
#data=data[0]


def convert(data):
    if isinstance(data, basestring):
        #return str(data)
         return data.encode('utf-8')
    elif isinstance(data, collections.Mapping):
        return dict(map(convert, data.iteritems()))
    elif isinstance(data, collections.Iterable):
        return type(data)(map(convert, data))
    else:
        return data
SS=0
rows = []
rows.append(['_id','event','track_id','id_session','browser','browser_version','city','country','ip','resolution','hostname','time_arrival','pages_seen','referrer','time_spent','media','created_at','updated_at','latitude','longitude','cpc','name']  )
f = open('shop_match.bson', 'rb')
stream = KeyValueBSONInput(fh=f)
for dict_data in stream:
    #print convert(dict_data)
    doc=convert(dict_data)
    A=doc['visitor']
  #print A 
    B=doc['result']
    C=doc['query']

    if ('event' in doc.keys() and '_id' in doc.keys() and 'created_at' in doc.keys() and 'updated_at' in doc.keys()   and  'brandName' in C.keys() and 'time_spent' in A.keys() and 'city' in A.keys()   and 'latitude' in A.keys() and 'longitude' in A.keys() and 'cpc' in B.keys() and 'name' in B.keys()): # and 'created_at_date' in doc.keys() and 'time_spent' in A.keys() and 'city'in A.keys() and 'track_id' in A.keys() and 'id_session' in A.keys() and  'zipcode' in A.keys()):# and 'time_arrival' in A.keys() and 'country'in A.keys() and 'hostname'in A.keys() and 'region' in  A.keys() and 'latitude' in A.keys() and 'longitude' in A.keys() and 'browser' in A.keys() and 'browser_version' in A.keys() and 'ip' in A.keys() and 'referrer'in A.keys() and 'pages_seen' in A.keys() and 'resolution' in A.keys()  and  'id' in B.keys() and 'name' in B.keys() and 'url' in B.keys() and 'is_spinoff' in B.keys() and 'cpc' in B.keys() and 'apikey' in C.keys()  and 'media' in C.keys() and 'src' in C.keys() and 'brandName' in C.keys() and 'contents' in C.keys()  and 'options' in C.keys() and 'formatOption' in C['options'].keys() and  'guiOption' in C['options'].keys()):
#rows.append([doc['_id'],doc['event'], A['track_id'],A['id_session'],A['browser'], A['browser_version'],A['city'],A['country'],A['ip'],A['resolution'],A['hostname'],A['time_arrival'],A['pages_seen'],A     ['referrer'],A['time_spent'],A['zipcode'],A['region'],A['latitude'],A['longitude'], B['id'], B['name'], B['url'],B['is_spinoff'], B['cpc'],C['apikey'],C['media'], C['options']['formatOption'],C['options']['guiOption'],C['brandName'] , C['contents'],doc['created_at'],doc['updated_at'], doc['created_at_date']] ):
      SS=SS+1
      rows.append([doc['_id'],doc['event'],A['track_id'],A['id_session'],A['browser'], A['browser_version'],A['city'],A ['country'],A['ip'],A['resolution'],A['hostname'],time.mktime(A['time_arrival'].timetuple()),A['pages_seen'],A['referrer'],A['time_spent'], C['media'],time.mktime(doc['created_at'].timetuple()),time.mktime(doc['updated_at'].timetuple()),A['latitude'],A['longitude'],B['cpc'],B['name'] ] )
   
with open('data_brut.csv', 'wb') as fp:
    a = csv.writer(fp, delimiter=',')
   
    a.writerows(rows)


f.close()

print ("data avaible")


