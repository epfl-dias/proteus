#!/bin/bash

expr_dir="/scratch2/sig20_pelago/expr_micro"
EXE_DIR="/scratch2/sig20_pelago/opt/pelago"



num_oltp_clients=28
num_olap_clients=16

cd $EXE_DIR

QUERY="Q1"


# # ETL
# for SF in 300
# do
# 	for qs in 1 2 4 6 8 10 12 14 16
# 	do
# 		for EXPR_NUM in {1..2}
# 		do
# 			qqq=$(expr $num_olap_clients / $qs)
# 			echo "#####################  $QUERY-LOCAL  ###########################"
# 			echo "#######################################################"
# 			echo "QueryPerSession: $qqq"
# 			echo "Number of Snapshots: $qs"
# 			echo "Experiment #: $EXPR_NUM"
# 			echo "#######################################################"
# 			echo "#######################################################"
# 			cmd_a="$EXE_DIR/htap-micro-local --etl=true --num-oltp-clients=$num_oltp_clients --num-olap-clients=$qs  --ch-scale-factor=$SF"
# 			echo "$cmd_a"
# 			$cmd_a 2>&1 | tee $expr_dir/micro-etl-$qqq-$SF-query_$EXPR_NUM

# 			kill $(ps aux | grep htap | awk '{print $2}')

# 			echo "Sleeping for 3 seconds before next expr."
# 			sleep 3
# 		done
# 	done
# done

# # REMOTE-QPI
# for SF in 300
# do
# 	for qs in 1 2 4 6 8 10 12 14 16
# 	do
# 		for EXPR_NUM in {1..2}
# 		do
# 			qqq=$(expr $num_olap_clients / $qs)
# 			echo "#####################  $QUERY-REMOTE  ###########################"
# 			echo "#######################################################"
# 			echo "QueryPerSession: $qqq"
# 			echo "Number of Snapshots: $qs"
# 			echo "Experiment #: $EXPR_NUM"
# 			echo "#######################################################"
# 			echo "#######################################################"
# 			cmd_a="$EXE_DIR/htap-micro-remote --num-oltp-clients=$num_oltp_clients --num-olap-clients=$qs  --ch-scale-factor=$SF"
# 			echo "$cmd_a"
# 			$cmd_a 2>&1 | tee $expr_dir/micro-remote-$qqq-$SF-query_$EXPR_NUM

# 			kill $(ps aux | grep htap | awk '{print $2}')

# 			echo "Sleeping for 3 seconds before next expr."
# 			sleep 3
# 		done
# 	done
# done

# # HYBRID
# for SF in 300
# do
# 	for qs in 1 2 4 6 8 10 12 14 16
# 	do
# 		for EXPR_NUM in {1..2}
# 		do
# 			qqq=$(expr $num_olap_clients / $qs)
# 			echo "#####################  $QUERY-HYBRID  ###########################"
# 			echo "#######################################################"
# 			echo "QueryPerSession: $qqq"
# 			echo "Number of Snapshots: $qs"
# 			echo "Experiment #: $EXPR_NUM"
# 			echo "#######################################################"
# 			echo "#######################################################"
# 			cmd_a="$EXE_DIR/htap-micro-hybrid --num-oltp-clients=$num_oltp_clients --num-olap-clients=$qs  --ch-scale-factor=$SF"
# 			echo "$cmd_a"
# 			$cmd_a 2>&1 | tee $expr_dir/micro-hybrid-$qqq-$SF-query_$EXPR_NUM

# 			kill $(ps aux | grep htap | awk '{print $2}')

# 			echo "Sleeping for 3 seconds before next expr."
# 			sleep 3
# 		done
# 	done
# done


# TRADE RESOURCES
for SF in 300
do
	# for qs in 1 2 4 6 8 10 12 14 16 18 20 22 24 26 28
	for qs in 12 16 20 24 28
	do
		for EXPR_NUM in {1..3}
		do
			qqq=$(expr $num_olap_clients / $qs)
			echo "#####################  EXCHANGE RESOURCES  ###########################"
			echo "#######################################################"
			echo "QueryPerSession: $qqq"
			echo "Number of Snapshots: $qs"
			echo "Experiment #: $EXPR_NUM"
			echo "#######################################################"
			echo "#######################################################"
			cmd_a="$EXE_DIR/htap-micro-remote --num-oltp-clients=$num_oltp_clients --num-olap-clients=1 --elastic=$qs --trade-core=true  --ch-scale-factor=$SF"
			echo "$cmd_a"
			$cmd_a 2>&1 | tee $expr_dir/micro-trade-$qs-$SF-query_$EXPR_NUM

			kill $(ps aux | grep htap | awk '{print $2}')

			echo "Sleeping for 3 seconds before next expr."
			sleep 3
		done
	done
done


#Elastic-Hybrid-Inserts
for SF in 300
do
	# for cr in 0 2 4 6 8 10 12 14 16 18 20 22 24 26 28
	for cr in 0 4 8 12 16 20 24 28
	do
		for qs in 1 2 4 # 6 8 10 12 14 16
		do
			for EXPR_NUM in {1..2}
			do
				start=`date +%s`
				qqq=$(expr $num_olap_clients / $qs)
				echo "#####################  Elastic-Hybrid-Inserts  ###########################"
				echo "#######################################################"
				echo "QueryPerSession: $qqq"
				echo "Number of Snapshots: $qs"
				echo "Experiment #: $EXPR_NUM"
				echo "#######################################################"
				echo "#######################################################"
				cmd_a="$EXE_DIR/htap-micro-hybrid --num-oltp-clients=$num_oltp_clients --elastic=$cr --num-olap-clients=$qs  --ch-scale-factor=$SF"
				echo "$cmd_a"
				$cmd_a 2>&1 | tee $expr_dir/micro-elastic-hybrid-insert-$qqq-$cr-$SF-query_$EXPR_NUM
				end=`date +%s`
				
				echo "Duration: $((($(date +%s)-$start)/60)) minutes"
				kill $(ps aux | grep htap | awk '{print $2}')

				echo "Sleeping for 3 seconds before next expr."
				sleep 3
			done
		done
	done
done

