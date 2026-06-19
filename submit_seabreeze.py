"""
Submit seabreeze_identification_v2.py
"""

import os
import subprocess
import time
import pandas as pd

def create_pbs_script(pbs_dir, pbs_filename, ncpus, mem, jobfsmem, queue, project, walltime, storage, name, command):
    pbs_script_content = f"""#!/bin/bash
  
#PBS -l ncpus={ncpus:d}
#PBS -l mem={mem:d}GB
#PBS -l jobfs={jobfsmem:d}GB
#PBS -q {queue}
#PBS -P {project}
#PBS -l walltime={walltime}
#PBS -l storage={storage}
#PBS -l wd
#PBS -o /scratch/up6/cx5009/PBS_output/
#PBS -e /scratch/up6/cx5009/PBS_output/
#PBS -M chang.xu8@unsw.edu.au
#PBS -m abe
#PBS -N {name}

{command}
"""
    with open(pbs_dir + pbs_filename, "w") as file:
        file.write(pbs_script_content)
    return pbs_dir+pbs_filename

def submit_job(env_vars, pbs_script, jobs_id):
    env_vars_str = ",".join([f"{key}={value}" for key, value in env_vars.items()])
    
    submit_command = ["qsub", "-v", env_vars_str, pbs_script]
    
    submission = subprocess.run(submit_command, check=True, stdout=subprocess.PIPE, text=True)
    submission_id = submission.stdout.strip()
    
    print(f"Submitted {pbs_script} as {submission_id}")
    
    jobs_id.append(submission_id)

def check_job_status(jobs_id, time_delay=10):
    print("\nWaiting for queue...\n")
    
    time.sleep(time_delay)
    
    print(f"Checking submission status at {time.ctime()}\n")
    
    for i in range(0, len(jobs_id)):
        try:
            check_command = ["qstat", "-swx", jobs_id[i]]
            
            checking = subprocess.run(check_command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            print(checking.stdout)
        except subprocess.CalledProcessError as e:
            print("Error:")
            print(e.stderr)

username = "cx5009"

#--------------------------------------------------------------------------------------------------
script_dir = "/g/data/up6/cx5009/hackathon/energy2026/"
script_filename = "seabreeze_identification_v3.py" 

# exp_seasons = ['SY_djf', 'SY_jja'] 
# exp_ress = ['SY_1', 'SY_5', 'SY_11p1']
# exp_ids = ['CTRL', 'NO-URBAN']

# test
# exp_seasons = ['SY_djf'] 
# exp_ress = ['SY_11p1'] 
# exp_ids = ['CTRL']

exp_seasons = ['SY_jja'] 
exp_ress = ['SY_11p1'] 
exp_ids = ['CTRL']

for exp_season in exp_seasons:
    for exp_res in exp_ress:
        for exp_id in exp_ids:

            if exp_season=='SY_djf':
                time_start = pd.to_datetime("2016-11-30 01:00:00")
                change_tlocal = 11
            else:
                time_start = pd.to_datetime("2017-05-31 01:00:00")
                change_tlocal = 10
                
            for time_now in range(18): # 18
                time_end = time_start + pd.Timedelta(5, "D") - pd.Timedelta(1, "h")
                
                env_vars = {}
                env_vars["EXP_SEASON"] = exp_season
                env_vars["EXP_RES"] = exp_res
                env_vars["EXP_ID"] = exp_id
                
                env_vars["T1"] = str(time_start)
                env_vars["T2"] = str(time_end)

                env_vars["LOCT"] = change_tlocal

                env_vars["TIME_PART"] = time_now
                
                pbs_dir = "/scratch/up6/cx5009/PBS_output/pbs_script/"
                ncpus = 28
                env_vars["NCPUS"] = ncpus
                mem = 256
                env_vars["MEM_GB"] = mem
                jobfsmem = 50
                queue = "normalbw"
                project = "nf33"
                walltime = "13:00:00"
                storage = "scratch/up6+gdata/up6+gdata/xp65+gdata/nf33+gdata/gb02"
                jobname = f'{exp_season}_{exp_res}_{exp_id}_{time_now}'
                
                jobs_id = []
                
                print(f"Start submitting job(s) at {time.ctime()}\n")
                
                pbs_filename = f'{exp_season}_{exp_res}_{exp_id}_{time_now}.sh'
                name = f"{exp_season}_{exp_res}_{exp_id}_{time_now}"
                command = f"""module use /g/data/xp65/public/modules
                module load conda/analysis3
                
                export BLOSC_NTHREADS=1
                export OMP_NUM_THREADS=1
                export OPENBLAS_NUM_THREADS=1
                export MKL_NUM_THREADS=1
                export NUMEXPR_MAX_THREADS=1
                export NUMEXPR_NUM_THREADS=1
                
                export TMPDIR=$PBS_JOBFS
                export DASK_TEMPORARY_DIRECTORY=$PBS_JOBFS
                export XDG_CACHE_HOME=$PBS_JOBFS/.cache
                mkdir -p "$TMPDIR" "$DASK_TEMPORARY_DIRECTORY" "$XDG_CACHE_HOME"
                python3 {script_dir}{script_filename}"""
                pbs_script = create_pbs_script(pbs_dir, pbs_filename, ncpus, mem, jobfsmem, queue, project, walltime, storage, name, command)
                    
                submit_job(env_vars, pbs_script, jobs_id)
                
                check_job_status(jobs_id)

                time_start = time_end + pd.Timedelta(1, "h")
            
