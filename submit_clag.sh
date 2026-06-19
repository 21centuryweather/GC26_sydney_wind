#!/bin/bash
#PBS -q normal
#PBS -P nf33
#PBS -l ncpus=24
#PBS -l mem=96GB
#PBS -l jobfs=1GB
#PBS -l walltime=08:00:00
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

python3 coastline_angle_cal.py



