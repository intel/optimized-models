num_cores=22
num_threads=3
COMMAND="$@"
for i in $(seq 1 $(($num_cores / $num_threads - 1)))
do
startid=$(($i*$num_threads))
endid=$(($i*$num_threads+$num_threads-1))
OMP_SCHEDULE=STATIC OMP_NUM_THREADS=$num_threads OMP_DISPLAY_ENV=TRUE OMP_PROC_BIND=TRUE GOMP_CPU_AFFINITY="$startid-$endid" KMP_AFFINITY=proclist=[$startid-$endid],granularity=thread,explicit $COMMAND &
done
wait
