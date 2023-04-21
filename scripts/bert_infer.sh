#/bin/bash

#OAR -p gpu='YES' and gpucapability>='7.0' and gpumem>='24000'
#OAR -t besteffort
#OAR -l /nodes=1/gpunum=1,walltime=48:00:00
#OAR --name train
#OAR --stdout outputs/%jobname%.%jobid%.out
#OAR --stderr outputs/%jobname%.%jobid%.err

module load conda/2021.11-python3.9
module load cuda/11.0
module load cudnn/8.0-cuda-11.0
module load gcc/7.3.0
source activate amazonml
cd /home/tdhamija/AmazonML/
python train.py
