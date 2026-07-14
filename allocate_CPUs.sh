#!/bin/bash
salloc --nodes $1 -n $2 --cpus-per-task=32 --ntasks-per-node=8 --qos premium --time 04:00:00 --constraint cpu --account m4872
