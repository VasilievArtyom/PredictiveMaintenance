#!/bin/bash

for tmstmp in {2330..2400}
do
python3 prediction.py $tmstmp
done
