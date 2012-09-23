#!/bin/python
from scandora import *
import scandora
import bregman as br
import pytagcrawl as T
import numpy as np
import itertools

def save_brain_features(ev_data=False, avg=True):
    
    subjects = ['1mar11sj','1mar11yw','5mar11ad','5mar11at','8mar11am','8mar11ec','9mar11ab','9mar11jd','16mar11hy','16mar11mg','16mar11mh','16mar11sg','17mar11sw','26feb11kj','26feb11zi']
    targets = '/global/data/casey/FMRI/pandora_dt_sm.labels.fixed'
    outpath = '/global/data/casey/thompson/brain_tags/brain_features/'
    for subject in subjects:
        print subject
        niibrain = '/global/data/casey/FMRI/'+subject+'.pandora_dt_sm.nii'
        mask = '/global/data/casey/FMRI/drawing/'+subject+'.resamp+sts+hg+orig.nii.gz'
        ds = load_fmri_data(niibrain, targets, ev_data=ev_data, mask=mask)
        ds = ds[ds.sa.targets < 6]
        if avg:
            # load stimuli labels
        stim_labels = '/global/data/casey/FMRI/stimuli_labels.txt'
        stimuli_ids = np.array([line.strip() for line in open(stim_labels)])
        stimuli_ids_tr = list(itertools.chain(*[[sid+'a', sid+'b', sid+'c'] for sid in stimuli_ids]))
            ds.sa['stim_ids'] = stimuli_ids_tr
            ds = np.array([np.mean(ds[ds.sa.stim_ids==i],0) for i in np.unique(stimuli_ids_tr)])
            fname = '_sts+hg_avgrun_75x72.brain'
        else:
            fname = '_sts+hg_600.brain'
        Brain, Vb, iVb = svd_reduce_data(ds,.99)
        bregman.audiodb.adb.write(outpath + subject + fname, Brain)
        
def save_tags():
    A = T.TrackTags() 
    A.tag_crawler()
    #A._run_track_top_tags()
    A._run_tag_tag_intersection()
    #A.unweighted_top_tags_to_feature_vectors('new_top100_tags_features.txt',100)
    A.weighted_top_tags_to_feature_vectors('weighted_top100_tags_features.txt',100)
    lastfm_feats = A.track_features
    pandora_feats = get_pandora_properties()


def get_pandora_properties():
    """
    Iterate through pandora properties files to generate feature vectors for each track. 
    Set all values in feature matrix to 0 or 100 to match probabilities of last.fm tag vectors.
    Save pandora matrix to file.
    Return pandora matrix to be concatenated to last.fm feature matrix.
    """
    unique_props = []
    #n_unique = 113
    #feature_mat = np.matrix()
    propdir = '/global/data/casey/FMRI/PandoraData/Properties/'
    for n,f in enumerate(TrackListA):
        print '**'
        print f
        properties = [line.strip() for line in open(propdir+'/'+f)]
        for m,p in enumerate(properties):
            if p in unique_props:
                i = unique_props.index(p)
                feature_mat[n,i] = 100
            else:
                unique_props.append(p)
                if n==0 and m==0:
                    feature_mat = np.zeros((25,1))
                    feature_mat[0,0] = 100
                else:
                    feature_mat = np.append(feature_mat, np.zeros((25,1)), 1)
                    feature_mat[n,-1] = 100
    # write features to file
    fname = 'pandora_feats.txt'
    if fname:
        try:
            np.savetxt(fname, feature_mat, fmt='%d')
        except:
            print "Could not save features to file %s\n" %(fname)
            raise
    return feature_mat

def save_weighted_and_pandora():
    """
    min number of tags=10, thus can evaluate p@<=10
    """
    lastfm_prob_feats = np.genfromtxt('weighted_top100_tags_features.txt')
    pandora_feats = get_pandora_properties()
    allfeats=np.append(lastfm_prob_feats, pandora_feats, 1)
    # write features to file
    fname = 'weighted_top100_lastfm+pandora_feats.txt'
    if fname:
        try:
            np.savetxt(fname, allfeats, fmt='%d')
        except:
            print "Could not save features to file %s\n" %(fname)
            raise


TrackListA=[
    "Brian Eno/brian+eno_a+clearing",
    "Brian Eno/various+artists+music+for+films+3_theme+from+creation",
    "Brian Eno/eno+moebius+roedelius_old+land",
    "Brian Eno/galerie+stratique_horizons+lointains",
    "Brian Eno/anugama_io+moon+of+jupiter",
    "Eddie Cochran/elvis+presley_jailhouse+rock",
    "Eddie Cochran/bill+haley_shake+rattle+and+roll",
    "Eddie Cochran/little+richard_bama+lama+bama+loo",
    "Eddie Cochran/ritchie+valens_come+on+lets+go",
    "Eddie Cochran/eddie+cochran_money+honey+live",
    "Ozzy Osbourne/ozzy+osbourne_fire+in+the+sky",
    "Ozzy Osbourne/judas+priest_youve+got+another+thing+comin",
    "Ozzy Osbourne/metallica_of+wolf+and+man",
    "Ozzy Osbourne/ac+dc_you+shook+me+all+night+long",
    "Ozzy Osbourne/scorpions_rock+you+like+a+hurricane",
    "Romantic/roger+norrington_sym+no9+in+d+op125+choral+2+molto+vivace+yvonne+kenny+sarah+walker+patrick+power+petteri+salomaa+schutz+chor+of+london_beethoven_symphony+number+9",
    "Romantic/colorado+symphony+orchestra_finale+allegro+con+fuoco_tchaikovsky_symphony+number+4",
    "Romantic/eugene+ormandy_3+vivacissimo_sibelius_symphony+number+2",
    "Romantic/royal+philharmonic+orchestra+schubert-symphonies+nos+3+5+6+beecham+royal+philharmonic+orchestra",
    "Romantic/roger+norrington_sym+no6+in+f+op68+pastoral+i+allegro+non+troppo+awakening+of+happy+feelings+on+arriving_beethoven_symphony+number+6",
    "Waylon Jennings/waylon+jennings_are+you+sure+hank+done+it+this+way",
    "Waylon Jennings/willie+nelson_me+and+paul",
    "Waylon Jennings/merle+haggard_pancho+and+lefty",
    "Waylon Jennings/hank+williams+junior_whiskey+bent+and+hell+bound",
    "Waylon Jennings/willie+nelson+waylon+jennings+johnny+cash+kris+kristofferson_welfare+line",
]

TrackListB=[
    ["Brian Eno", "Brian Eno", "A Clearing "],
    ["Brian Eno", "Brian Eno", "Theme from 'Creation' "],
    ["Brian Eno", "Eno, Moebius & Roedelius", "Old Land "],
    ["Brian Eno", "Galerie Stratique", "Horizons Lointains "],
    ["Brian Eno", "Anugama", "IO-Moon of Jupiter "],
    ["Eddie Cochran", "Elvis Presley", "Jailhouse Rock "],
    ["Eddie Cochran", "Bill Haley", "Shake Rattle and Roll "],
    ["Eddie Cochran", "Little Richard", "Bama Lama Bama Loo "],
    ["Eddie Cochran", "Ritchie Valens", "Come On Let's Go "],
    ["Eddie Cochran", "Eddie Cochran", "Money Honey "],
    ["Ozzy Osbourne", "Ozzy Osbourne", "Fire in the Sky "],
    ["Ozzy Osbourne", "Judas Priest", "You've Got Another Thing Coming "],
    ["Ozzy Osbourne", "Metallica", "Of Wolf & Man "],
    ["Ozzy Osbourne", "ACDC", "You Shook Me All Night Long "],
    ["Ozzy Osbourne", "Scorpions", "Rock You Like A Hurricane "],
    ["Romantic Symphony", "Beethoven", "Symphony No. 9 Mvt. 2 "],
    ["Romantic Symphony", "Tchaikovsky", "Symphony No. 4 Mvt. 4 "],
    ["Romantic Symphony", "Sibelius", "Symphony No. 2 Mvt. 4 "],
    ["Romantic Symphony", "Schubert", "Symphony No. 5 Mvt. 1 "],
    ["Romantic Symphony", "Beethoven", "Symphony No. 6 Mvt. 1 "],
    ["Waylon Jennings", "Waylon Jennings", "Are You Sure Hank Done It This Way "],
    ["Waylon Jennings", "Willie Nelson", "Me and Paul "],
    ["Waylon Jennings", "Merle Haggard", "Pancho and Lefty "],
    ["Waylon Jennings", "Hank Williams Jr", "Whiskey Bent and Hell Bound "],
    ["Waylon Jennings", "Willie Nelson", "Welfare Line "],
]