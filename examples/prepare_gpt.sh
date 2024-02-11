#! /bin/bash
#$ -l rt_C.small=1
#$ -l h_rt=24:00:00
#$ -j y
#$-cwd
# source /etc/profile.d/modules.sh
# module load python/3.10/3.10.10
# source ~/GSLM/venv/bin/activate
type=$1
layer=$2
python3 ./prepare_for_gpt_tts_libriheavy.py  \
  ulm.librilight.manifest_root=/scratch/acc12576tt/librilight/output/ \
  ulm.librilight.manifest_pattern="libriheavy_cuts_*.jsonl.gz" \
  ulm.librilight.feature_root=/scratch/acc12576tt/librilight/fairseq_manifest \
  ulm.librilight.feature_pattern=libriheavy_cuts_\*.txt_"$type"_quantized_l"$layer"_km1000.feature \
  ulm.librilight.output_jsonl_path=/home/acc12576tt/scratch/gpt_neox_data/"$type"_l"$layer".jsonl \
