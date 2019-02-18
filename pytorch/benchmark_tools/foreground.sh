num_cores=22
num_threads=3
COMMAND="$@"
endid=$(($num_threads-1))
OMP_SCHEDULE=STATIC OMP_NUM_THREADS=$num_threads OMP_DISPLAY_ENV=TRUE OMP_PROC_BIND=TRUE GOMP_CPU_AFFINITY="0-$endid" KMP_AFFINITY=proclist=[0-$endid],granularity=thread,explicit $COMMAND 
