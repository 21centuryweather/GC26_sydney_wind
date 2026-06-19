#!/bin/bash

for season in SY_djf SY_jja 
do
    for res in SY_11p1
    do
        for exp in NO-URBAN
        do

            qsub \
                -N ${season}_${res}_${exp} \
                -v EXP_SEASON=${season},EXP_RES=${res},EXP_ID=${exp} \
                submit_sbi.sh

        done
    done
done



