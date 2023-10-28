#PJM -L rscgrp=share-debug
#PJM -L gpu=1
#PJM -L elapse=0:10:00
#PJM -g ge43
#PJM -j
module load gcc/8.3.1
module load python/3.8.12
source venv/bin/activate
python3 train_u2s.py u2s/datamodule=wisteria_jvs