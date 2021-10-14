#!/bin/bash
#grep -r '^[^(Socket)]TPS.*MTPS' . | sort
expr_dir="/scratch/raza/gc-sz-try"
MODE="oneshot"
RUNTIME=30
DELTA_SIZE=10

echo "Hostname connected: $HOSTNAME"

if [ "$HOSTNAME" = "diascld33" ]; then
 # dias33
   EXE_DIR="/tmp/tmp.KNX8E8mZRE/cmake-build-dias33/apps/benchmarks/oltp"
   WORK_DIR="/tmp/tmp.KNX8E8mZRE/cmake-build-dias33/opt/pelago"
   WORKER_THREADS=(1 2 4 8 16 32 36 64 72 90 108 126 128 144)
elif [ "$HOSTNAME" = "iccluster149" ]; then
 # ic149
     EXE_DIR="/tmp/tmp.W8gnWscJyR/cmake-build-ic149/apps/benchmarks/oltp"
     WORK_DIR="/tmp/tmp.W8gnWscJyR/cmake-build-ic149/opt/pelago"
     # WORKER_THREADS=(1 2 4 6 8 10 12 14 16 18 20 22 24 26 28) # 32 36 40 44 48 52 56)
     # WORKER_THREADS=(1 4 8 12 14 16 20 24 28 32 36 40 44 48 52 56) # 32 36 40 44 48 52 56)
     WORKER_THREADS=(1 4 8 12 14 16 20 24 28)
     # WORKER_THREADS=(1 4 8 12 14 16 20 24 28)

elif [ "$HOSTNAME" = "diascld46" ]; then
 # diascld46
     EXE_DIR="/tmp/tmp.57XzMbXvqo/cmake-build-dias46/apps/benchmarks/oltp"
     WORK_DIR="/tmp/tmp.57XzMbXvqo/cmake-build-dias46/opt/pelago"
     WORKER_THREADS=(1 2 4 8 12 16 24 36 48)
else
 echo "Unknown target machine: $HOSTNAME"
 exit
fi




cd $WORK_DIR || exit

LD_LIBRARY_PATH="../lib:/scratch/pelago/latest/opt/lib"
export LD_LIBRARY_PATH

# YCSB ZIPF
#
for N_WORKER in 28 14
do
    for RW_RATIO in 100 50
    do
      for N_COLS in 1
      do
        for SIZE in 1 2 3 4 5 6 7 8 9 10
        do

         for EXPR_NUM in 1 2
         do
           ZIPF=50
           RW_RATIO_ACTUAL=$(bc <<<"scale=2; $RW_RATIO / 100")
           ZIPF_ACTUAL=$(bc <<<"scale=2; $ZIPF / 100")
           NUM_REC=$( expr 1000 '*' "$N_WORKER")
           start=`date +%s`
           echo "##################### YCSB-WriteOnly  ###########################"
           echo "Experiment #: $EXPR_NUM"
           echo "N_WORKER: $N_WORKER"
           echo "#######################################################"

           cmd_a="$EXE_DIR/oltp-bench-runner  --ycsb-num-ops-per-txn=$SIZE --ycsb-zipf-theta=$ZIPF_ACTUAL --ycsb-num-records=$NUM_REC --ycsb-num-cols=$N_COLS --ycsb_write_ratio=$RW_RATIO_ACTUAL --delta-size=$DELTA_SIZE --num-workers=$N_WORKER --runtime=$RUNTIME --benchmark=0"
           echo "$cmd_a"
           $cmd_a 2>&1 | tee $expr_dir/ycsbSize-$MODE-RW-$RW_RATIO_ACTUAL-ZIPF-$ZIPF_ACTUAL-cols-$N_COLS-nRec-$NUM_REC-nWrkr-$N_WORKER-Nops-$SIZE-Nexpr-$EXPR_NUM
           # end=`date +%s`

           echo "Duration: $((($(date +%s)-$start)/60)) minutes"
           kill $(ps aux | grep oltp-bench-runner | awk '{print $2}')

           echo "Sleeping for 2 seconds before next expr."
           sleep 2
         done
       done
   done
 done
done

