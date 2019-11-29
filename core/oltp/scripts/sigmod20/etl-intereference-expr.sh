#!/bin/bash

expr_dir="/scratch/raza/cidr20/experiments"

EXE_DIR="/scratch/raza/cidr20/opt/pelago"
PLAN_DIR="/scratch/raza/cidr20/src/executor/inputs/plans/sigmod20/htap/micro_bench"

ETL_PLAN="plan_etl_scan.json"


num_oltp_clients=36
num_olap_clients=16

cd $EXE_DIR


for qs in 16 8 4 2 1
do
	for EXPR_NUM in {1..5}
	do
		qqq=$(expr $num_olap_clients / $qs)
		echo "#######################################################"
		echo "#######################################################"
		echo "QueryPerSession: $qqq"
		echo "Number of Snapshots: $qs"
		echo "Experiment #: $EXPR_NUM"
		echo "#######################################################"
		echo "#######################################################"
		#cow_query_expr_100="/scratch/raza/cidr20/opt/pelago/htap-server-fork-100-snapshot  --num-oltp-clients=36 --num-olap-clients=36 --plan-json=/scratch/raza/cidr20/plan.json"
		cmd_a="$EXE_DIR/htap-server-circular-master --etl=true --num-oltp-clients=$num_oltp_clients --num-olap-clients=$qs --plan-json=$PLAN_DIR/$ETL_PLAN"
		cmd="$cmd_a | tee $expr_dir/htap-server-etl-$qqq-query_$EXPR_NUM"
		$cmd_a | tee $expr_dir/htap-server-etl-$qqq-query_$EXPR_NUM

		kill $(ps aux | grep htap | awk '{print $2}')

		echo "Sleeping for 10 seconds before next expr."
		sleep 10
	done
done
