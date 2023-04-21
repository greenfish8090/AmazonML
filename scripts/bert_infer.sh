#/bin/bash

#OAR -p gpu='YES' and gpucapability>='7.0' and gpumem>='24000'
#OAR -t besteffort
#OAR -l /nodes=1/gpunum=1,walltime=48:00:00
#OAR --name train
#OAR --stdout outputs/%jobname%.%jobid%.out
#OAR --stderr outputs/%jobname%.%jobid%.err

echo "==================RUN SCRIPT=================="
echo "$(cat scripts/bert_infer.sh)"
echo -e "==================RUN SCRIPT==================\n"
module load cuda/10.2
module load cudnn/7.6-cuda-10.2
module load gcc/6.2.0

source activate amazonmlnew
python --version
cat $OAR_NODEFILE
cat $OAR_RESOURCE_PROPERTIES_FILE

nvidia-smi

python bert_inference.py --batch_size 128 --num_workers 0

nvidia-smi
