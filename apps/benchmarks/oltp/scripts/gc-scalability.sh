#!/bin/bash
#grep -r '^[^(Socket)]TPS.*MTPS' . | sort

expr_dir="/scratch/raza/gc3/oneshot-scale"
MODE="oneshot"
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
      WORKER_THREADS=(1 2 4 8 14 16 28 32 42 56)
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

## YCSB SCALE
for N_WORKER in "${WORKER_THREADS[@]}";
do
  NUM_COLS=10
  NUM_REC=$( expr 1000000 '*' "$N_WORKER")
  for EXPR_NUM in 1 2 3
  do
    start=`date +%s`
    echo "##################### YCSB-WriteOnly  ###########################"
    echo "Experiment #: $EXPR_NUM"
    echo "N_WORKER: $N_WORKER"
    echo "#######################################################"

    cmd_a="$EXE_DIR/oltp-bench-runner --ycsb-num-records=$NUM_REC --ycsb-num-cols=$NUM_COLS --ycsb_write_ratio=1.0 --delta-size=$DELTA_SIZE --num-workers=$N_WORKER --runtime=$RUNTIME --benchmark=0"
    echo "$cmd_a"
    $cmd_a 2>&1 | tee $expr_dir/ycsb-writeOnly-$MODE-cols-$NUM_COLS-numRec-$NUM_REC-numWorkers-$N_WORKER-exprNum-$EXPR_NUM
#    end=`date +%s`

    echo "Duration: $((($(date +%s)-$start)/60)) minutes"
    kill $(ps aux | grep oltp-bench-runner | awk '{print $2}')

    echo "Sleeping for 2 seconds before next expr."
    sleep 2
  done
done

# YCSB-ReadWrite (0.5 ratio)
for N_WORKER in "${WORKER_THREADS[@]}"
do
  NUM_COLS=10
  NUM_REC=$( expr 1000000 '*' "$N_WORKER")
  for EXPR_NUM in 1 2 3
  do
    start=`date +%s`
    echo "##################### YCSB-WriteOnly  ###########################"
    echo "Experiment #: $EXPR_NUM"
    echo "N_WORKER: $N_WORKER"
    echo "#######################################################"

    cmd_a="$EXE_DIR/oltp-bench-runner --ycsb-num-records=$NUM_REC --ycsb-num-cols=$NUM_COLS --ycsb_write_ratio=0.5 --delta-size=$DELTA_SIZE --num-workers=$N_WORKER --runtime=$RUNTIME --benchmark=0"
    echo "$cmd_a"
    $cmd_a 2>&1 | tee $expr_dir/ycsb-readWrite-$MODE-cols-$NUM_COLS-numRec-$NUM_REC-numWorkers-$N_WORKER-exprNum-$EXPR_NUM
#    end=`date +%s`

    echo "Duration: $((($(date +%s)-$start)/60)) minutes"
    kill $(ps aux | grep oltp-bench-runner | awk '{print $2}')

    echo "Sleeping for 2 seconds before next expr."
    sleep 2
  done
done

#TPCC
for N_WORKER in "${WORKER_THREADS[@]}"
do
  for EXPR_NUM in 1 2 3
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


# YCSB R/W RATIOS

#for N_COLS in 10
#do
#  for N_WORKER in  28 56 #"${WORKER_THREADS[@]}"
#  do
#    NUM_REC=$( expr 1000000 '*' "$N_WORKER")
#    for RW_RATIO in 0 20 40 50 60 80 100
#    do
#      RW_RATIO_ACTUAL=$(bc <<<"scale=2; $RW_RATIO / 100")
#      for ZIPF in 50
#      do
#        ZIPF_ACTUAL=$(bc <<<"scale=2; $ZIPF / 100")
#          for EXPR_NUM in 1 2 # 3
#          do
#            start=`date +%s`
#            echo "##################### YCSB-WriteOnly  ###########################"
#            echo "Experiment #: $EXPR_NUM"
#            echo "N_WORKER: $N_WORKER"
#            echo "#######################################################"
#
#            cmd_a="$EXE_DIR/oltp-bench-runner --ycsb-zipf-theta=$ZIPF_ACTUAL --ycsb-num-records=$NUM_REC --ycsb-num-cols=$N_COLS --ycsb_write_ratio=$RW_RATIO_ACTUAL --delta-size=$DELTA_SIZE --num-workers=$N_WORKER --runtime=$RUNTIME --benchmark=0"
#            echo "$cmd_a"
#            $cmd_a 2>&1 | tee $expr_dir/ycsbRw-RW-$RW_RATIO_ACTUAL-ZIPF-$ZIPF_ACTUAL-$MODE-cols-$N_COLS-nRec-$NUM_REC-nWrkr-$N_WORKER-Nexpr-$EXPR_NUM
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

# YCSB ZIPF

for N_COLS in 10
do
  for N_WORKER in 56 28 14 42  #"${WORKER_THREADS[@]}"
  do
    NUM_REC=$( expr 1000000 '*' "$N_WORKER")
    for RW_RATIO in 0 20 40 50 60 80 100 #50
    do
      RW_RATIO_ACTUAL=$(bc <<<"scale=2; $RW_RATIO / 100")
      for ZIPF in 0 25 50 75 90 99
      do
        ZIPF_ACTUAL=$(bc <<<"scale=2; $ZIPF / 100")
          for EXPR_NUM in 1 2 #3
          do
            start=`date +%s`
            echo "##################### YCSB-WriteOnly  ###########################"
            echo "Experiment #: $EXPR_NUM"
            echo "N_WORKER: $N_WORKER"
            echo "#######################################################"

            cmd_a="$EXE_DIR/oltp-bench-runner --ycsb-zipf-theta=$ZIPF_ACTUAL --ycsb-num-records=$NUM_REC --ycsb-num-cols=$N_COLS --ycsb_write_ratio=$RW_RATIO_ACTUAL --delta-size=$DELTA_SIZE --num-workers=$N_WORKER --runtime=$RUNTIME --benchmark=0"
            echo "$cmd_a"
            $cmd_a 2>&1 | tee $expr_dir/ycsbZipf-RW-$RW_RATIO_ACTUAL-ZIPF-$ZIPF_ACTUAL-$MODE-cols-$N_COLS-nRec-$NUM_REC-nWrkr-$N_WORKER-Nexpr-$EXPR_NUM
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