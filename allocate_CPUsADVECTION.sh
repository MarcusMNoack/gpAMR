#!/bin/bash
salloc -N $1 -n $2 --ntasks-per-node=4 --cpus-per-task=32 --qos premium --time 04:00:00 --constraint cpu --account m4872
