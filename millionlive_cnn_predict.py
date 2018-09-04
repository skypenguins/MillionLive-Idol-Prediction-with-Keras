import sys, os.path, glob
import json
from collections import OrderedDict
from keras.models import Model, load_model
from keras.utils import plot_model
import numpy as np

millionlive_dir = './'
script_dir = millionlive_dir + 'script_data/'
idol_meta_data_file = millionlive_dir + 'millionlive_idolname.json'
tmp_dir = millionlive_dir + 'tmp/'

# jsonの読み込み
def load_json(idol_meta_data_file=idol_meta_data_file):
    with open(idol_meta_data_file, 'r') as f:
        idol_meta_data = json.load(f, object_pairs_hook=OrderedDict) # jsonの順序を保持したままOrderdDictを生成
        #print(idol_meta_data['idols'])
        for idol in idol_meta_data['idols']:
            print('{0}\tid: {1}\tidol_name: {2}'.format(idol['idol_id'], idol['id'], idol['idol_name']))
    return idol_meta_data['idols']

def encode_pred_text(raw_txt):
    txt = [ord(x) for x in str(raw_txt).strip().replace('　','')]
    txt = txt[:200]
    if len(txt) < 200:
        txt += ([0] * (200 - len(txt)))
    return txt

def predict(pred_txt, model_filepath='model.h5'):
    model = load_model(model_filepath)
    plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)
    for line in pred_txt:
        _txt = encode_pred_text(line)
        result = model.predict(np.array([_txt]))
    return result

if __name__ == '__main__':
    idols = load_json()
    predict_results = predict(sys.stdin)[0,:]
    sorted_results = sorted([(i,e) for i,e in enumerate(list(predict_results))], key=lambda x:x[1]*-1)
    #print(sorted_results)
    for result in sorted_results:
        id, prob = result
        if float(prob)*100 == 0: 
            prob += .01
        print('{}\t: {}%'.format(idols[id]['idol_name'], round(float(prob)*100, 2)))
