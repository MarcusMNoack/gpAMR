#!/bin/bash
salloc --nodes 4 -n 16 --cpus-per-task=32 --ntasks-per-node=4 --qos interactive --time 04:00:00 --constraint cpu --account m4872
