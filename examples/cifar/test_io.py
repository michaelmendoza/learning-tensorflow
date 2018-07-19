
'''
file = 'results.txt'
historyList = ['1','2']
import _pickle as cPickle # For python 3
fo = open(file, 'wb')
cPickle.dump(historyList, fo)
fo.close()

fo = open(file, 'rb')
dict = cPickle.load(fo, encoding='latin-1')
print(dict)
'''

import json
obj = ['1','2']
file = 'results.json'
with open(file, 'w') as outfile:
    json.dump(obj, outfile)
    print(json.dumps(obj, indent=4))

with open(file) as json_data:
    d = json.load(json_data)
    print(d)