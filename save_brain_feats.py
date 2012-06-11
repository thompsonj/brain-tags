#!/bin/python
from scandora import *
import scandora
import bregman as br
import pytagcrawl as T

def save_brain_features():
    subjects = ['1mar11sj','1mar11yw','5mar11ad','5mar11at','8mar11am','8mar11ec','9mar11ab','9mar11jd','16mar11hy','16mar11mg','16mar11mh','16mar11sg','17mar11sw','26feb11kj','26feb11zi']
    targets = '/global/data/casey/FMRI/pandora_dt_sm.labels.fixed'
    outpath = '/global/data/casey/thompson/brain_tags/brain_features/'
    for subject in subjects:
        niibrain = '/global/data/casey/FMRI/'+subject+'.pandora_dt_sm.nii'
        mask = '/global/data/casey/FMRI/drawing/'+subject+'.resamp+sts+orig.nii.gz'
        ds = load_fmri_data(niibrain, targets, ev_data=True, mask=mask)
        ds = ds[ds.sa.targets < 6]
        Brain, Vb, iVb = svd_reduce_data(ds.S, 0.99)
        bregman.audiodb.adb.write(outpath + subject + '_200_99.brain', Brain)
        
def save_tags():
    A = T.TrackTags() 
    A._run_track_top_tags()
    A._run_tag_tag_intersection()
    A.unweighted_top_tags_to_feature_vectors('new_top100_tags_features.txt',100)
