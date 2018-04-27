#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --job-name="training"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=2GB
#SBATCH --time=12:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sb6416@nyu.edu
#SBATCH --output="training-%j.out"
#SBATCH --err="training-%j.err"

module purge
#module load jupyter-kernels/py2.7
#module load jupyter-kernels/py3.5

module load cuda/9.0.176
module load cudnn/9.0v7.0.5
module load python3/intel/3.6.3

source /home/sb6416/pyenv/py3.6.3/bin/activate



SRCDIR=$HOME/language-model-bias/model
OUTPUT=/beegfs/sb6416/ds1012/models
 

python $SRCDIR/training_from_txt.py --cuda --dropout 0.5 --epoch 40 --nhid 1500 --emsize 1500 --lr 20 --tied --bias_reg --bias_reg_factor 0.2 --bias_reg_var_ratio 0.5 --save $OUTPUT/model_dm_1.pt

python $SRCDIR/training_from_txt.py --cuda --dropout 0.5 --epoch 10 --nhid 300 --emsize 300 --lr 20 --tied --bias_reg --bias_reg_factor 0.5 --bias_reg_var_ratio 0.5 --save $OUTPUT/model_dm_2.pt
