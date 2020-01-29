#!/bin/bash

expr_dir="/scratch1/sig20_rev_1/"
EXE_DIR="/scratch1/pelago/opt/pelago"
SEQ_COUNT=20
QUERY="MIX"
ELASTIC_THRESH=4

cd $EXE_DIR

#ISOLATED
for SF in 1
do
		for EXPR_NUM in {1..3}
		do
			MODE="ISOLATED"			
			start=`date +%s`
			echo "#####################  ISOLATED  ###########################"
			echo "#######################################################"
			echo "PerQueryFreshness: True"
			echo "Experiment #: $EXPR_NUM"
			echo "#######################################################"
			echo "#######################################################"
			cmd="./htap-ch-bench --ch-scale-factor=$SF --htap-mode=$MODE --num-olap-repeat=$SEQ_COUNT"
			echo "$cmd"
			$cmd_a 2>&1 | tee $expr_dir/htap-$MODE-queryFresh-sf-$SF-$QUERY_$EXPR_NUM
			end=`date +%s`
			
			echo "Duration: $((($(date +%s)-$start)/60)) minutes"
			kill $(ps aux | grep htap | awk '{print $2}')

			echo "Sleeping for 3 seconds before next expr."
			sleep 3
		done
		
done


#COLOCATED
for SF in 1
do
		for EXPR_NUM in {1..3}
		do
			MODE="COLOCATED"			
			start=`date +%s`
			echo "#####################  COLOCATED  ###########################"
			echo "#######################################################"
			echo "PerQueryFreshness: True"
			echo "Experiment #: $EXPR_NUM"
			echo "#######################################################"
			echo "#######################################################"
			cmd="./htap-ch-bench --ch-scale-factor=$SF --htap-mode=$MODE --num-olap-repeat=$SEQ_COUNT"
			echo "$cmd"
			$cmd_a 2>&1 | tee $expr_dir/htap-$MODE-queryFresh-sf-$SF-$QUERY_$EXPR_NUM
			end=`date +%s`
			
			echo "Duration: $((($(date +%s)-$start)/60)) minutes"
			kill $(ps aux | grep htap | awk '{print $2}')

			echo "Sleeping for 3 seconds before next expr."
			sleep 3
		done
		
done

#S3_IS
for SF in 1
do
		for ETL in 0 250 500 1000
		do
			for EXPR_NUM in {1..3}
			do
				MODE="HYBRID-ISOLATED"			
				start=`date +%s`
				echo "#####################  HYBRID-ISOLATED  ###########################"
				echo "#######################################################"
				echo "PerQueryFreshness: True"
				echo "Experiment #: $EXPR_NUM"
				echo "#######################################################"
				echo "#######################################################"
				cmd="./htap-ch-bench --ch-scale-factor=$SF --htap-mode=$MODE --num-olap-repeat=$SEQ_COUNT --etl-interval-ms=$ETL"
				echo "$cmd"
				$cmd_a 2>&1 | tee $expr_dir/htap-$MODE-queryFresh-sf-$SF-ETL-$ETL-$QUERY_$EXPR_NUM
				end=`date +%s`
				
				echo "Duration: $((($(date +%s)-$start)/60)) minutes"
				kill $(ps aux | grep htap | awk '{print $2}')

				echo "Sleeping for 3 seconds before next expr."
				sleep 3
			done
		done
done

#S3_NI
for SF in 1
do
	for ETL in 0 250 500 1000
		do
			for EXPR_NUM in {1..3}
			do
				MODE="HYBRID-COLOC"			
				start=`date +%s`
				echo "#####################  HYBRID-COLOC  ###########################"
				echo "#######################################################"
				echo "PerQueryFreshness: True"
				echo "Experiment #: $EXPR_NUM"
				echo "#######################################################"
				echo "#######################################################"
				cmd="./htap-ch-bench --ch-scale-factor=$SF --htap-mode=$MODE --num-olap-repeat=$SEQ_COUNT --oltp-elastic-threshold=$ELASTIC_THRESH  --etl-interval-ms=$ETL"
				echo "$cmd"
				$cmd_a 2>&1 | tee $expr_dir/htap-$MODE-queryFresh-sf-$SF-ETL-$ETL-$QUERY_$EXPR_NUM
				end=`date +%s`
				
				echo "Duration: $((($(date +%s)-$start)/60)) minutes"
				kill $(ps aux | grep htap | awk '{print $2}')

				echo "Sleeping for 3 seconds before next expr."
				sleep 3
			done
		done
done


#ADAPTIVE
# for SF in 1
# do
# 		for EXPR_NUM in {1..3}
# 		do
# 			MODE="ADAPTIVE"			
# 			start=`date +%s`
# 			echo "#####################  ADAPTIVE  ###########################"
# 			echo "#######################################################"
# 			echo "PerQueryFreshness: True"
# 			echo "Experiment #: $EXPR_NUM"
# 			echo "#######################################################"
# 			echo "#######################################################"
# 			cmd="./htap-ch-bench --ch-scale-factor=$SF --htap-mode=$MODE --num-olap-repeat=$SEQ_COUNT --oltp-elastic-threshold=$ELASTIC_THRESH"
# 			echo "$cmd"
# 			$cmd_a 2>&1 | tee $expr_dir/htap-$MODE-queryFresh-sf-$SF-$QUERY_$EXPR_NUM
# 			end=`date +%s`
			
# 			echo "Duration: $((($(date +%s)-$start)/60)) minutes"
# 			kill $(ps aux | grep htap | awk '{print $2}')

# 			echo "Sleeping for 3 seconds before next expr."
# 			sleep 3
# 		done
		
# done



#-------------


#ISOLATED
for SF in 1
do
		for EXPR_NUM in {1..3}
		do
			MODE="ISOLATED"			
			start=`date +%s`
			echo "#####################  ISOLATED  ###########################"
			echo "#######################################################"
			echo "PerQueryFreshness: False"
			echo "Experiment #: $EXPR_NUM"
			echo "#######################################################"
			echo "#######################################################"
			cmd="./htap-ch-bench --ch-scale-factor=$SF --htap-mode=$MODE --num-olap-repeat=$SEQ_COUNT --per-query-snapshot=false"
			echo "$cmd"
			$cmd_a 2>&1 | tee $expr_dir/htap-$MODE-sf-$SF-$QUERY_$EXPR_NUM
			end=`date +%s`
			
			echo "Duration: $((($(date +%s)-$start)/60)) minutes"
			kill $(ps aux | grep htap | awk '{print $2}')

			echo "Sleeping for 3 seconds before next expr."
			sleep 3
		done
		
done


#COLOCATED
for SF in 1
do
		for EXPR_NUM in {1..3}
		do
			MODE="COLOCATED"			
			start=`date +%s`
			echo "#####################  COLOCATED  ###########################"
			echo "#######################################################"
			echo "PerQueryFreshness: False"
			echo "Experiment #: $EXPR_NUM"
			echo "#######################################################"
			echo "#######################################################"
			cmd="./htap-ch-bench --ch-scale-factor=$SF --htap-mode=$MODE --num-olap-repeat=$SEQ_COUNT --per-query-snapshot=false"
			echo "$cmd"
			$cmd_a 2>&1 | tee $expr_dir/htap-$MODE-sf-$SF-$QUERY_$EXPR_NUM
			end=`date +%s`
			
			echo "Duration: $((($(date +%s)-$start)/60)) minutes"
			kill $(ps aux | grep htap | awk '{print $2}')

			echo "Sleeping for 3 seconds before next expr."
			sleep 3
		done
		
done

#S3_IS
for SF in 1
do
		for ETL in 0 250 500 1000
		do
			for EXPR_NUM in {1..3}
			do
				MODE="HYBRID-ISOLATED"			
				start=`date +%s`
				echo "#####################  HYBRID-ISOLATED  ###########################"
				echo "#######################################################"
				echo "PerQueryFreshness: False"
				echo "Experiment #: $EXPR_NUM"
				echo "#######################################################"
				echo "#######################################################"
				cmd="./htap-ch-bench --ch-scale-factor=$SF --htap-mode=$MODE --num-olap-repeat=$SEQ_COUNT --etl-interval-ms=$ETL --per-query-snapshot=false"
				echo "$cmd"
				$cmd_a 2>&1 | tee $expr_dir/htap-$MODE-sf-$SF-ETL-$ETL-$QUERY_$EXPR_NUM
				end=`date +%s`
				
				echo "Duration: $((($(date +%s)-$start)/60)) minutes"
				kill $(ps aux | grep htap | awk '{print $2}')

				echo "Sleeping for 3 seconds before next expr."
				sleep 3
			done
		done
done

#S3_NI
for SF in 1
do
	for ETL in 0 250 500 1000
		do
			for EXPR_NUM in {1..3}
			do
				MODE="HYBRID-COLOC"			
				start=`date +%s`
				echo "#####################  HYBRID-COLOC  ###########################"
				echo "#######################################################"
				echo "PerQueryFreshness: False"
				echo "Experiment #: $EXPR_NUM"
				echo "#######################################################"
				echo "#######################################################"
				cmd="./htap-ch-bench --ch-scale-factor=$SF --htap-mode=$MODE --num-olap-repeat=$SEQ_COUNT --oltp-elastic-threshold=$ELASTIC_THRESH  --etl-interval-ms=$ETL --per-query-snapshot=false"
				echo "$cmd"
				$cmd_a 2>&1 | tee $expr_dir/htap-$MODE-sf-$SF-ETL-$ETL-$QUERY_$EXPR_NUM
				end=`date +%s`
				
				echo "Duration: $((($(date +%s)-$start)/60)) minutes"
				kill $(ps aux | grep htap | awk '{print $2}')

				echo "Sleeping for 3 seconds before next expr."
				sleep 3
			done
		done
done

