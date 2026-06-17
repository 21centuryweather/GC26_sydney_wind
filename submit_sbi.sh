#!/bin/bash
#PBS -q normal
#PBS -P k10
#PBS -l ncpus=24
#PBS -l mem=96GB
#PBS -l jobfs=1GB
#PBS -l walltime=08:00:00
#PBS -l storage=scratch/k10+gdata/k10+gdata/xp65+gdata/nf33+gdata/gb02
#PBS -l wd
#PBS -o /g/data/k10/ds1182/seabreeze_work/PBS_output/
#PBS -e /g/data/k10/ds1182/seabreeze_work/PBS_output/
#PBS -M danny.shadrech@monash.edu
#PBS -m abe


cd /g/data/k10/ds1182/GC26_sydney_wind

module purge
module use /g/data/xp65/public/modules
module load conda/analysis3

python3 /g/data/k10/ds1182/seabreeze_work/seabreeze_identification.py \
    ${EXP_SEASON} \
    ${EXP_RES} \
    ${EXP_ID}









