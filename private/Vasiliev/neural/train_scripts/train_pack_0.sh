#!/bin/bash

for blc_id in 4 7 #{0..5} 
do
for steps in {1..20}
do
python3 trainmodels.py $blc_id $steps 0
done
done
