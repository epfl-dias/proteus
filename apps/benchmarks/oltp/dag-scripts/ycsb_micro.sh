#!/bin/bash

expr_dir="/scratch/raza/dag-expr/new"
EXE_DIR="/tmp/tmp.gYUL9ljBdR/cmake-build-dias-33/apps/benchmarks/oltp"
WORK_DIR="/tmp/tmp.gYUL9ljBdR/cmake-build-dias-33/opt/pelago"

MODE="full-list-record"
# --num-workers=36 --ycsb-zipf-theta=0.5 --benchmark=0 --runtime=30  --ycsb-write-ratio=0.5 --ycsb-num-cols=16 --ycsb-num-col-upd=8 --ycsb-num-col-read=8 --ycsb-num-col-read-offset=8 --delta-size=4


NUM_WORKERS=36
ZIPF=0.75
RUNTIME=30
DELTA_SIZE=6
RW=0.5
NUM_COLS=32

COL_UPD=8
COL_READ=8

cd $WORK_DIR

LD_LIBRARY_PATH="../lib:/scratch/pelago/latest/opt/lib"
export LD_LIBRARY_PATH



## WriteOnly
#for wCol in 1 2 4 8 16 32 64
#do
#
#			for EXPR_NUM in {1..2}
#			do
#				start=`date +%s`
#				echo "#####################  YCSB-WRITE-Only  ###########################"
#				echo "#######################################################"
#				echo "Experiment #: $EXPR_NUM"
#				echo "WriteColumns (Total: $NUM_COLS) #: $wCol"
#				echo "#######################################################"
#				echo "#######################################################"
#
#				cmd_a="$EXE_DIR/oltp-bench-runner --num-workers=$NUM_WORKERS --ycsb-zipf-theta=$ZIPF --benchmark=0 --runtime=$RUNTIME --delta-size=$DELTA_SIZE --ycsb-write-ratio=1 --ycsb-num-cols=$NUM_COLS --ycsb-num-col-upd=$wCol --ycsb-num-col-read=0"
#				echo "$cmd_a"
#				$cmd_a 2>&1 | tee $expr_dir/ycsb-$MODE-write-only-$NUM_COLS-write-$wCol-$EXPR_NUM
#				#htap-server-hybrid-insert-$qqq-elastic-MemMove-$cr-$SF-query_$EXPR_NUM
#				end=`date +%s`
#
#				echo "Duration: $((($(date +%s)-$start)/60)) minutes"
#				kill $(ps aux | grep oltp-bench-runner | awk '{print $2}')
#
#				echo "Sleeping for 3 seconds before next expr."
#				sleep 2
#			done
#done



# OverLap
for OFFSET in 8 4 0
do

			for EXPR_NUM in {1..2}
			do
				start=`date +%s`
				echo "#####################  YCSB-WRITE-Only  ###########################"
				echo "#######################################################"
				echo "Experiment #: $EXPR_NUM"
				echo "WriteColumns (Total: $NUM_COLS) #: $wCol"
				echo "#######################################################"
				echo "#######################################################"

        # --num-workers=36 --ycsb-zipf-theta=0.5 --benchmark=0 --runtime=30 --delta-size=6 --ycsb-write-ratio=0.5 --ycsb-num-cols=16 --ycsb-num-col-upd=8 --ycsb-num-col-read=8 --ycsb-num-col-read-offset=0

				cmd_a="$EXE_DIR/oltp-bench-runner --num-workers=$NUM_WORKERS --ycsb-zipf-theta=$ZIPF --benchmark=0 --runtime=$RUNTIME --delta-size=$DELTA_SIZE --ycsb-write-ratio=$RW --ycsb-num-cols=$NUM_COLS --ycsb-num-col-upd=$COL_UPD --ycsb-num-col-read=$COL_READ --ycsb-num-col-read-offset=$OFFSET"
				echo "$cmd_a"
				$cmd_a 2>&1 | tee $expr_dir/ycsb-$MODE-overlap-only-$NUM_COLS-offset-$OFFSET-$EXPR_NUM
				#htap-server-hybrid-insert-$qqq-elastic-MemMove-$cr-$SF-query_$EXPR_NUM
				end=`date +%s`

				echo "Duration: $((($(date +%s)-$start)/60)) minutes"
				kill $(ps aux | grep oltp-bench-runner | awk '{print $2}')

				echo "Sleeping for 3 seconds before next expr."
				sleep 2
			done
done

