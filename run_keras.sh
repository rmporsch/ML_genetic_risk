#!/bin/bash
#PBS -l nodes=1:ppn=10 
#PBS -l mem=50gb 
#PBS -l walltime=24:00:00 
#PBS -q medium
#PBS -N keras_nn

script='/home/rmporsch/projects/ML_genetic_risk/nnpredict/kkpredict.py'

python $script 'nn' -p 5
