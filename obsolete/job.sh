#!/bin/bash
#SBATCH -A m1516_g
#SBATCH -C gpu
##SBATCH -A m1516
##SBATCH -C cpu
#SBATCH -q regular
##SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -n 4
#SBATCH --ntasks-per-node=4
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=map_gpu:0,1,2,3,0,1,2,3
##SBATCH --nodes=2
##SBATCH --ntasks-per-node=1
#SBATCH -J gpAMR64
#SBATCH -e %j.err
#SBATCH -o %j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=treb@lbl.gov

export CH_TIMER=1
export CH_OUTPUT_INTERVAL=10000
export MPICH_GPU_SUPPORT_ENABLED=0

out_dir="striped"
if [ ! -d $out_dir ] ; then
  mkdir $out_dir
  stripe_large $out_dir
fi

input="flowpastcylinder2d.inputs"

#ProgName=/global/homes/u/u6338/_svn_2025-04-24/svn_treb/EBAMRINS/execGPAMR/gpamrDriver2d.Linux.64.g++.gfortran.OPTHIGH.MPI.ex
ProgName=/global/homes/u/u6338/_svn_2025-09-12/svn_treb/EBAMRINS/execGPAMR/gpamrDriver2d.Linux.64.g++.gfortran.OPTHIGH.MPI.ex

export OMP_NUM_THREADS=128

#module load PrgEnv-gnu cpe-cuda

srun -C gpu -n 4 -G 4 --exclusive $ProgName $input >& log
#srun -C cpu -n 2 --exclusive $ProgName $input >& log
