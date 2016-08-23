# -*- coding: utf-8 -*-

from sklearn.externals import joblib
import numpy as np

dicts = joblib.load("../../data/dict_figer")

a = joblib.load("atts.pkl")
atts, scores = np.array(a[0])[:,:,0].T, a[1]

for i,score in enumerate(scores):
    print ", ".join([str(v) for v in list(atts[i])])+"\t",
    lid,ls = max(enumerate(list(score)),key=lambda x: x[1])
    print dicts["id2label"][lid], ls,
    for label_id,label_score in enumerate(list(score)):
        if label_score > 0.5:
            if label_id != lid:
                print dicts["id2label"][label_id], label_score, 
    print



