#!/bin/bash
#SBATCH -A m5044_g
#SBATCH -C gpu
#SBATCH -q premium
#SBATCH -t 02:00:00
#SBATCH -N 2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=map_gpu:0,1,2,3


rm -f log

malloc_trim_threshold_=0
module load python/3.11
module load craype-x86-milan
module load libfabric/1.22.0
module load craype-network-ofi
module load xpmem/2.9.7-1.1_20250411150514__g191b5f8bea4c
module load PrgEnv-gnu/8.5.0
module load cray-dsmml/0.3.0
module load cray-libsci/24.07.0
module load cray-mpich/8.1.30
module load craype/2.7.32
module load gcc-native/13.2
module load perftools-base/24.07.0
module load cpe/24.07
module load cudatoolkit/12.4
module load craype-accel-nvidia80
module load gpu/1.0
module load sqs/2.0
module load darshan/default

echo "setting up Chombo run"
export MPICH_GPU_SUPPORT_ENABLED=0
# progname="gpamrDriver2d.Linux.64.g++.gfortran.OPTHIGH.MPI.ex"
progname="viscousDriver2d.Linux.64.g++.gfortran.OPTHIGH.MPI.ex"


# input="flowpastcylinder2d.inputs"
input="inclusion.inputs"

srun --exact -C gpu --nodes=2 --ntasks=8 $progname $input


