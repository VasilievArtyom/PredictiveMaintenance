#!/bin/bash

for blc_id in {1..9} 
do
for steps in {20..1}
do
fixedstep=$((steps + 1))
#echo $blc_id $steps $fixedstep
mv ${blc_id}_binary_on_${steps}.txt ${blc_id}_binary_on_${fixedstep}.txt
done
done
