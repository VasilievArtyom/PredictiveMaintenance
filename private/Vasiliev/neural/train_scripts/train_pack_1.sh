#!/bin/bash

for blc_id in {8..9} 
do
for steps in {1..20}
do
python3 trainmodels.py $blc_id $steps 1
done
done
