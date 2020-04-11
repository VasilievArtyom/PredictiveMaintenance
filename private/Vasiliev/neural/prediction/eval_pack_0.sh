#!/bin/bash

for blc_id in {0..9} 
do

python3 evaluate_models.py $blc_id 0
done
