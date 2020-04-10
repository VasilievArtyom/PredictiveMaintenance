#!/bin/bash

for blc_id in {0..9} 
do
for pred_stps in {1..21}
do
python3 plot_evaluated.py $blc_id $pred_stps
done
done
