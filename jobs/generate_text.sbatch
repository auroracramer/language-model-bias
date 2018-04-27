#!/usr/bin/env bash
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
OUTPUT=/beegfs/sb6416/ds1012/gen_text_bbc
MOD=/beegfs/jtc440/ds1012
 

python $SRCDIR/generate_from_txt.py --cuda --words 10000 --seed 1234 --checkpoint $MOD/model_bias_reg.pt --outf $OUTPUT/generated_man_bbc.txt

python $SRCDIR/generate_from_txt.py --cuda --words 10000 --seed 5678 --checkpoint $MOD/model_bias_reg.pt --outf $OUTPUT/generated_man_bbc.txt


