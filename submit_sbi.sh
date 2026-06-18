#!/bin/bash

#PBS -q normal
#PBS -P nf33
#PBS -l ncpus=48
#PBS -l mem=190GB
#PBS -l jobfs=1GB
#PBS -l walltime=12:00:00
#PBS -l storage=scratch/up6+gdata/up6+gdata/xp65+gdata/nf33+gdata/gb02
#PBS -l wd
#PBS -o /scratch/up6/cx5009/PBS_output/
#PBS -e /scratch/up6/cx5009/PBS_output/
#PBS -M chang.xu8@unsw.edu.au
#PBS -m abe


cd /g/data/up6/cx5009/hackathon/energy2026

module purge
module use /g/data/xp65/public/modules
module load conda/analysis3

echo "START $(date)"
echo "$EXP_SEASON $EXP_RES $EXP_ID"

python3 -u seabreeze_identification_v2.py \
    ${EXP_SEASON} \
    ${EXP_RES} \
    ${EXP_ID}

echo "END $(date)"









