#!/bin/bash

export SLURM_CPU_BIND="cores"
scheduler_file=$SCRATCH/scheduler_filegpAMR.json
rm -f $scheduler_file

MALLOC_TRIM_THRESHOLD_=0
module load python/3.11-24.1.0
source ./gpAMRenv/bin/activate
export PYTHONPATH=$PWD:$PYTHONPATH
echo we have nodes: ${SLURM_JOB_NODELIST}

unset SRUN_CPUS_PER_TASK

hn=$(hostname -s)
port="8788"
echo ${port}
echo "starting scheduler"
export DASK_DISTRIBUTED__COMM__TIMEOUTS__CONNECT=3600s
export DASK_DISTRIBUTED__COMM__TIMEOUTS__TCP=3600s
export DASK_DISTRIBUTED__SCHEDULER__WORK_STEALING=False
export DASK_DISTRIBUTED__SCHEDULER__WORKER_SATURATION=1
MPICH_GPU_SUPPORT_ENABLED=0

dask scheduler --no-dashboard \
    --interface hsn0 \
    --scheduler-file $scheduler_file &

dask_pid=$!
# Wait for the scheduler to start
sleep 1
until [ -f $scheduler_file ]
do
     sleep 1
done
until grep -q '"address"' $scheduler_file 2>/dev/null; do sleep 1; done


echo "starting workers"
DASK_DISTRIBUTED__COMM__TIMEOUTS__CONNECT=3600s
DASK_DISTRIBUTED__COMM__TIMEOUTS__TCP=3600s


srun -N 1 -n 2 --ntasks-per-node=2 --cpus-per-task=32 \
     -o dask_worker_%t.txt -e dask_worker_%t.err dask worker --memory-limit="30 GiB" \
    --scheduler-file $scheduler_file \
    --interface hsn0 \
    --no-dashboard \
    --nthreads 32 \
    --nworkers 1







