#!/bin/bash

for blc_id in {6..9} 
do

python3 evaluate_models.py $blc_id 1
done
