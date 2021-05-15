import requests
import json

test_sample = json.dumps({'data': [
    [2015-08-28,1302556,N,ES,V,39,2014-09-05,0.0,12,1.0,,1,A,S,N,,KFC,N,1.0,36.0,PONTEVEDRA,1.0,,02 - PARTICULARES], 
    [2016-03-28,907940,N,ES,V,40,2011-03-24,0.0,60,1.0,,1,I,S,N,,KFC,N,1.0,28.0,MADRID,0.0,90220.62,02 - PARTICULARES]
]})
test_sample = str(test_sample)

def test_ml_service(scoreurl, scorekey):
    assert scoreurl != None

    if scorekey is None:
        headers = {'Content-Type':'application/json'}
    else:
        headers = {'Content-Type':'application/json', 'Authorization':('Bearer ' + scorekey)}

    resp = requests.post(scoreurl, test_sample, headers=headers)
    assert resp.status_code == requests.codes.ok
    assert resp.text != None
    assert resp.headers.get('content-type') == 'application/json'
    assert int(resp.headers.get('Content-Length')) > 0
