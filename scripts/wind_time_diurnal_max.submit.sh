#!/bin/bash

#PBS -l ncpus=1
#PBS -l mem=100GB
#PBS -l jobfs=1GB
#PBS -q normal
#PBS -P nf33
#PBS -l walltime=01:00:00
#PBS -l storage=gdata/xp65+gdata/gb02+scratch/gb02
#PBS -o /home/565/mr4682/GC26_sydney_wind/scripts/wind_time_diurnal_max_SY_SY_1_NO-URBAN-v1-201705310100-201709010000.out
#PBS -e /home/565/mr4682/GC26_sydney_wind/scripts/wind_time_diurnal_max_SY_SY_1_NO-URBAN-v1-201705310100-201709010000.err

cd /home/565/mr4682/GC26_sydney_wind

module purge
module use /g/data/xp65/public/modules
module load conda/analysis3

python3 /home/565/mr4682/GC26_sydney_wind/scripts/wind_time_diurnal_max.py