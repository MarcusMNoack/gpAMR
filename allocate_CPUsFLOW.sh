#!/bin/bash
salloc --nodes $1 -n $2 --qos premium --time 04:00:00 --constraint cpu --account m1516
