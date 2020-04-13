#!/bin/bash

for pred_stps in {1..21}
do
python3 thresholds.py $1 $pred_stps
done
