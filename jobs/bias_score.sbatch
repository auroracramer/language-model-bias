#!/usr/bin/env bash

#SBATCH --job-name=bias_score
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64GB
#SBATCH --time=7-0
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sb6416@nyu.edu
#SBATCH --output="bias_score-%j.out"
#SBATCH --err="bias_score-%j.err"

#source ~/.bashrc

source /home/sb6416/pyenv/py3.6.3/bin/activate


SRCDIR=$HOME/language-model-bias




DATASET_DIR=/beegfs/yw3004/projects/language_bias/generated_sentences/bbc
OUTPUT_DIR=/beegfs/sb6416/ds1012/scores/dm



module purge

python $SRCDIR/bias_score.py $DATASET_DIR $OUTPUT_DIR -w 7 -n 1
