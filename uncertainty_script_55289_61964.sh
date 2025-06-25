#PBS -N xubo_revision
#PBS -P 11004044
#PBS -l select=1:ncpus=96:mem=256gb
#PBS -j oe
#PBS -m ea
#PBS -V
#PBS -l walltime=24:00:00

# Change to the working directory
cd $PBS_O_WORKDIR
echo "Working directory is $PBS_O_WORKDIR"

# Define the log file path
LOG_FILE="task_log.txt"
echo "Task log will be written to: $LOG_FILE"
echo "Task log started at $(date)" > $LOG_FILE

# Activate the virtual environment
source activate prep-shot

# Task parameters
start_id=55289
end_id=61964
max_parallel_tasks=10
task_counter=0

# Dynamic parallel task management
for ((i=start_id; i<=end_id; i++)); do
    # If the number of running tasks reaches the maximum, wait for one task to complete
    if [ $task_counter -ge $max_parallel_tasks ]; then
        wait -n  # Wait for any background task to complete
        task_counter=$((task_counter - 1))  # Decrement the task counter
    fi

    # Check if the output file already exists
    if [ -f "output/Uncertainty/mekong_2020_288h_4_years_AD110dams__${i}.nc" ]; then
        echo "ID $i: File already exists. Skipping." | tee -a $LOG_FILE
        sync  # Force flush the file system buffer
    else
        echo "ID $i: Starting task..." | tee -a $LOG_FILE
        sync  # Force flush the file system buffer
        # Run the task and log the result
        (
            python run_mekong_gurobi-uncertainty.py $i $i
            if [ $? -eq 0 ]; then
                echo "ID $i: Task completed successfully." | tee -a $LOG_FILE
            else
                echo "ID $i: Task failed!" | tee -a $LOG_FILE
            fi
            sync  # Force flush the file system buffer
        ) &
        task_counter=$((task_counter + 1))  # Increment the task counter
    fi
done

# Wait for all remaining background tasks to complete
wait
echo "All tasks completed." | tee -a $LOG_FILE
echo "Task log ended at $(date)" >> $LOG_FILE
sync  # Force flush the file system buffer
