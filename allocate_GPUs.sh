#!/bin/bash
salloc --nodes $1 -n $2 --gpus-per-task=1 --qos premium --time 04:00:00 --constraint gpu -G $2 --account m5044_g
