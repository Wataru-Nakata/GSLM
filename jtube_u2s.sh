#!/bin/bash

#$ -l rt_F=1
#$ -l h_rt=72:0:00
#$ -j y
#$ -cwd
source /etc/profile.d/modules.sh
source .venv/bin/activate
cd examples
python train_u2s.py u2s=xvector u2s/datamodule=jtube_v100_xvector.yaml
