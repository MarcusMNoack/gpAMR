#!/bin/bash

./cleanup.sh
scancel $(cat jobid.txt)
sleep 5
jobid=$(sbatch jobscript.sh | awk '{print $4}')
echo $jobid
echo $jobid > jobid.txt


