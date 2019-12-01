#!/bin/bash

expr_dir="/scratch2/sig20_pelago/exper"

EXE_DIR="/scratch2/sig20_pelago/opt/pelago"
PLAN_DIR="/scratch2/sig20_pelago/htap/ch-plans/"

#ETL_PLAN="q1-local.json"
#ELASTIC_PLAN="q1-elastic.json"
#REMOTE_PLAN="q1-remote.json"

#ETL_PLAN="q16-modified-local.json"
#ELASTIC_PLAN="q16-modified-elastic.json"
#REMOTE_PLAN="q16-modified-remote.json"


ETL_PLAN="msum-local.json"
ELASTIC_PLAN="msum-remote.json"
REMOTE_PLAN="msum-elastic.json"


num_oltp_clients=28
num_olap_clients=16

cd $EXE_DIR




# ELASTIC
# for qs in 4 1
# do
# 	for EXPR_NUM in {1..3}
# 	do
# 		qqq=$(expr $num_olap_clients / $qs)
# 		echo "#####################  Q1-ELASTIC  ###########################"
# 		echo "#######################################################"
# 		echo "QueryPerSession: $qqq"
# 		echo "Number of Snapshots: $qs"
# 		echo "Experiment #: $EXPR_NUM"
# 		echo "#######################################################"
# 		echo "#######################################################"
# 		#cow_query_expr_100="/scratch/raza/cidr20/opt/pelago/htap-server-fork-100-snapshot  --num-oltp-clients=36 --num-olap-clients=36 --plan-json=/scratch/raza/cidr20/plan.json"
# 		cmd_a="$EXE_DIR/htap-server-circular-master --num-oltp-clients=$num_oltp_clients --elastic=true --num-olap-clients=$qs --plan-json=$PLAN_DIR/$ELASTIC_PLAN"
# 		cmd="$cmd_a | tee $expr_dir/htap-server-elastic-$qqq-query_$EXPR_NUM"
# 		$cmd_a | tee $expr_dir/htap-server-elastic-$qqq-query_$EXPR_NUM

# 		kill $(ps aux | grep htap | awk '{print $2}')

# 		echo "Sleeping for 5 seconds before next expr."
# 		sleep 5
# 	done
# done

# LOCAL
for qs in 16 8 4 2 1
do
	# for EXPR_NUM in {1..3}
	# do
		qqq=$(expr $num_olap_clients / $qs)
		echo "#####################  Q1-LOCAL  ###########################"
		echo "#######################################################"
		echo "QueryPerSession: $qqq"
		echo "Number of Snapshots: $qs"
		echo "Experiment #: $EXPR_NUM"
		echo "#######################################################"
		echo "#######################################################"
		#cow_query_expr_100="/scratch/raza/cidr20/opt/pelago/htap-server-fork-100-snapshot  --num-oltp-clients=36 --num-olap-clients=36 --plan-json=/scratch/raza/cidr20/plan.json"
		cmd_a="$EXE_DIR/htap-server-circular-master --num-oltp-clients=$num_oltp_clients --num-olap-clients=$qs --etl   --ch-scale-factor=$SF"
		cmd="$cmd_a | tee $expr_dir/htap-server-local-$qqq-query_$EXPR_NUM"
		echo "$cmd_a"
		$cmd_a 2>&1 | tee $expr_dir/htap-server-etl-sf100-$qqq-query_$EXPR_NUM

		kill $(ps aux | grep htap | awk '{print $2}')

		echo "Sleeping for 5 seconds before next expr."
		sleep 10
	# done
done

# # REMOTE
# for qs in 4 1
# do
# 	for EXPR_NUM in {1..3}
# 	do
# 		qqq=$(expr $num_olap_clients / $qs)
# 		echo "#####################  Q1-REMOTE  ###########################"
# 		echo "#######################################################"
# 		echo "QueryPerSession: $qqq"
# 		echo "Number of Snapshots: $qs"
# 		echo "Experiment #: $EXPR_NUM"
# 		echo "#######################################################"
# 		echo "#######################################################"
# 		#cow_query_expr_100="/scratch/raza/cidr20/opt/pelago/htap-server-fork-100-snapshot  --num-oltp-clients=36 --num-olap-clients=36 --plan-json=/scratch/raza/cidr20/plan.json"
# 		cmd_a="$EXE_DIR/htap-server-circular-master --num-oltp-clients=$num_oltp_clients --num-olap-clients=$qs --plan-json=$PLAN_DIR/$REMOTE_PLAN"
# 		cmd="$cmd_a | tee $expr_dir/htap-server-remote-$qqq-query_$EXPR_NUM"
# 		$cmd_a | tee $expr_dir/htap-server-remote-$qqq-query_$EXPR_NUM

# 		kill $(ps aux | grep htap | awk '{print $2}')

# 		echo "Sleeping for 5 seconds before next expr."
# 		sleep 5
# 	done
# done

# # ETL
# for qs in 4 1
# do
# 	for EXPR_NUM in {1..3}
# 	do
# 		qqq=$(expr $num_olap_clients / $qs)
# 		echo "#####################  Q1-ETL  ###########################"
# 		echo "#######################################################"
# 		echo "QueryPerSession: $qqq"
# 		echo "Number of Snapshots: $qs"
# 		echo "Experiment #: $EXPR_NUM"
# 		echo "#######################################################"
# 		echo "#######################################################"
# 		#cow_query_expr_100="/scratch/raza/cidr20/opt/pelago/htap-server-fork-100-snapshot  --num-oltp-clients=36 --num-olap-clients=36 --plan-json=/scratch/raza/cidr20/plan.json"
# 		cmd_a="$EXE_DIR/htap-server-circular-master --etl=true --num-oltp-clients=$num_oltp_clients --num-olap-clients=$qs --plan-json=$PLAN_DIR/$ETL_PLAN"
# 		cmd="$cmd_a | tee $expr_dir/htap-server-etl-$qqq-query_$EXPR_NUM"
# 		$cmd_a | tee $expr_dir/htap-server-etl-$qqq-query_$EXPR_NUM

# 		kill $(ps aux | grep htap | awk '{print $2}')

# 		echo "Sleeping for 5 seconds before next expr."
# 		sleep 10
# 	done
# done







