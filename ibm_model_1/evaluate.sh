#!/bin/bash

# Evaluates alignment from FILE
# using different tresholds

printf "tres\tAER\tprec\trec\n"
for treshold in $(seq 0 0.1 1); do
    #treshold=`echo $treshold "/ 10" | bc -l`
    printf "$treshold\t"
    paste <(./filter_treshold.py $1 $treshold) czenali | cut -f 1,4,5 | ./alignment-error-rate.pl 2>/dev/null
done