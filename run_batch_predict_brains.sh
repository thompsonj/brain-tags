#!/bin/bash

WALLTIME=048:00:00
DIR=/global/data/casey/thompson/brain_tags
SUBJECTS="1mar11yw 
5mar11ad 
5mar11at 
8mar11am 
8mar11ec 
9mar11ab 
9mar11jd 
16mar11hy 
16mar11mg 
16mar11mh 
16mar11sg 
17mar11sw 
26feb11kj 
26feb11zi"
#1mar11sj"
# KS="1
# 5
# 10"
KS="5"
# feats_size="600"

for subject in $SUBJECTS
  do
      for k in $KS
      do
          proc=$$
          SUBJECT=${subject/*\/}
          cat $DIR/predict_brains.pbs | sed "s/SUBJECT/$SUBJECT/" | sed "s/WALLTIME/$WALLTIME/" > /tmp/predict_brains.$proc
          echo $subject $k
          qsub -v subject=$subject /tmp/predict_brains.$proc
      done
  done

