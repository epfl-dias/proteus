#!/bin/bash
#grep -r '^[^(Socket)]TPS.*MTPS' . | sort
expr_dir="/scratch/raza/gc7"
MODE="steam"
RUNTIME=60
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

#TPCC
for N_WORKER in "${WORKER_THREADS[@]}"
do
 for EXPR_NUM in 1 2
 do
   start=`date +%s`
   echo "##################### TPCC  ###########################"
   echo "Experiment #: $EXPR_NUM"
   echo "N_WORKER: $N_WORKER"
   echo "#######################################################"

   cmd_a="$EXE_DIR/oltp-bench-runner --delta-size=$DELTA_SIZE --cpu-buffers=0.01 --tpcc-num-wh=$N_WORKER --num-workers=$N_WORKER --runtime=$RUNTIME --benchmark=1"
   echo "$cmd_a"
   $cmd_a 2>&1 | tee $expr_dir/tpcc-$MODE-numWorkers-$N_WORKER-exprNum-$EXPR_NUM
#    end=`date +%s`

   echo "Duration: $((($(date +%s)-$start)/60)) minutes"
   kill $(ps aux | grep oltp-bench-runner | awk '{print $2}')

   echo "Sleeping for 2 seconds before next expr."
   sleep 2
 done
done

# YCSB ZIPF
#
#for N_COLS in 10 1
#do
# for ZIPF in 50 #0 90 99
# do
#   for RW_RATIO in 50 100 #50
#   do
#     for N_WORKER in "${WORKER_THREADS[@]}"
#     do
#         for EXPR_NUM in 1 2
#         do
#           RW_RATIO_ACTUAL=$(bc <<<"scale=2; $RW_RATIO / 100")
#           ZIPF_ACTUAL=$(bc <<<"scale=2; $ZIPF / 100")
#           NUM_REC=$( expr 1000 '*' "$N_WORKER")
#           start=`date +%s`
#           echo "##################### YCSB-WriteOnly  ###########################"
#           echo "Experiment #: $EXPR_NUM"
#           echo "N_WORKER: $N_WORKER"
#           echo "#######################################################"
#
#           cmd_a="$EXE_DIR/oltp-bench-runner --ycsb-zipf-theta=$ZIPF_ACTUAL --ycsb-num-records=$NUM_REC --ycsb-num-cols=$N_COLS --ycsb_write_ratio=$RW_RATIO_ACTUAL --delta-size=$DELTA_SIZE --num-workers=$N_WORKER --runtime=$RUNTIME --benchmark=0"
#           echo "$cmd_a"
#           $cmd_a 2>&1 | tee $expr_dir/ycsbZipf-$MODE-RW-$RW_RATIO_ACTUAL-ZIPF-$ZIPF_ACTUAL-cols-$N_COLS-nRec-$NUM_REC-nWrkr-$N_WORKER-Nexpr-$EXPR_NUM
#           # end=`date +%s`
#
#           echo "Duration: $((($(date +%s)-$start)/60)) minutes"
#           kill $(ps aux | grep oltp-bench-runner | awk '{print $2}')
#
#           echo "Sleeping for 2 seconds before next expr."
#           sleep 2
#         done
#       done
#   done
# done
#done


# YCSB - NumColumns
# YCSB - NumWorkers
# YCSB - WriteRatio
# YCSB - Zipfian

# THE BIG LOOP
#for N_COLS in 1 2 4 8 10 16
#do
#  for N_WORKER in "${WORKER_THREADS[@]}"
#  do
#    NUM_REC=$( expr 1000000 '*' "$N_WORKER")
#    for RW_RATIO in 0 25 50 75 100
#    do
#      RW_RATIO_ACTUAL=$(bc <<<"scale=2; $RW_RATIO / 100")
#      for ZIPF in 0 25 50 75
#      do
#        ZIPF_ACTUAL=$(bc <<<"scale=2; $ZIPF / 100")
#          for EXPR_NUM in 1 2 3
#          do
#            start=`date +%s`
#            echo "##################### YCSB-WriteOnly  ###########################"
#            echo "Experiment #: $EXPR_NUM"
#            echo "N_WORKER: $N_WORKER"
#            echo "#######################################################"
#
#            cmd_a="$EXE_DIR/oltp-bench-runner --ycsb-zipf-theta=$ZIPF_ACTUAL --ycsb-num-records=$NUM_REC --ycsb-num-cols=$N_COLS --ycsb_write_ratio=$RW_RATIO_ACTUAL --delta-size=$DELTA_SIZE --num-workers=$N_WORKER --runtime=$RUNTIME --benchmark=0"
#            echo "$cmd_a"
#            $cmd_a 2>&1 | tee $expr_dir/ycsb-RW-$RW_RATIO_ACTUAL-ZIPF-$ZIPF_ACTUAL-$MODE-cols-$N_COLS-nRec-$NUM_REC-nWrkr-$N_WORKER-Nexpr-$EXPR_NUM
#            # end=`date +%s`
#
#            echo "Duration: $((($(date +%s)-$start)/60)) minutes"
#            kill $(ps aux | grep oltp-bench-runner | awk '{print $2}')
#
#            echo "Sleeping for 2 seconds before next expr."
#            sleep 2
#          done
#        done
#    done
#  done
#done