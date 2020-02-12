#!/bin/bash

expr_dir="/scratch1/sig20_rev_feb12/"
EXE_DIR="/scratch1/pelago_two/opt/pelago"
#SEQ_COUNT=150
QUERY="MIX"
ELASTIC_THRESH=4

cd $EXE_DIR

for SEQ_COUNT in 110
do

	#ISOLATED
	for SF in 30
	do
			for EXPR_NUM in {1..1}
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
				$cmd 2>&1 | tee $expr_dir/htap-$MODE-queryFresh-sf-$SF-$QUERY-SEQ-COUNT-$SEQ_COUNT-$EXPR_NUM
				end=`date +%s`
				
				echo "Duration: $((($(date +%s)-$start)/60)) minutes"
				kill $(ps aux | grep htap | awk '{print $2}')

				echo "Sleeping for 3 seconds before next expr."
				sleep 3
			done
			
	done


	# #COLOCATED
	# for SF in 30
	# do
	# 		for EXPR_NUM in {1..1}
	# 		do
	# 			MODE="COLOC"			
	# 			start=`date +%s`
	# 			echo "#####################  COLOCATED  ###########################"
	# 			echo "#######################################################"
	# 			echo "PerQueryFreshness: True"
	# 			echo "Experiment #: $EXPR_NUM"
	# 			echo "#######################################################"
	# 			echo "#######################################################"
	# 			cmd="./htap-ch-bench --ch-scale-factor=$SF --htap-mode=$MODE --num-olap-repeat=$SEQ_COUNT"
	# 			echo "$cmd"
	# 			$cmd 2>&1 | tee $expr_dir/htap-$MODE-queryFresh-sf-$SF-$QUERY-SEQ-COUNT-$SEQ_COUNT-$EXPR_NUM
	# 			end=`date +%s`
				
	# 			echo "Duration: $((($(date +%s)-$start)/60)) minutes"
	# 			kill $(ps aux | grep htap | awk '{print $2}')

	# 			echo "Sleeping for 3 seconds before next expr."
	# 			sleep 3
	# 		done
			
	# done

	#READ OVER QPI
	for SF in 30
	do
		for EXPR_NUM in {1..1}
		do
			MODE="REMOTE-READ"			
			start=`date +%s`
			echo "#####################  HYBRID-ISOLATED  ###########################"
			echo "#######################################################"
			echo "PerQueryFreshness: True"
			echo "Experiment #: $EXPR_NUM"
			echo "#######################################################"
			echo "#######################################################"
			cmd="./htap-ch-bench --ch-scale-factor=$SF --htap-mode=$MODE --num-olap-repeat=$SEQ_COUNT"
			echo "$cmd"
			$cmd 2>&1 | tee $expr_dir/htap-$MODE-queryFresh-sf-$SF-ETL-$ETL-$QUERY-SEQ-COUNT-$SEQ_COUNT-$EXPR_NUM
			end=`date +%s`
			
			echo "Duration: $((($(date +%s)-$start)/60)) minutes"
			kill $(ps aux | grep htap | awk '{print $2}')

			echo "Sleeping for 3 seconds before next expr."
			sleep 3
		done
	done

	#S3_IS
	for SF in 30
	do
			for ETL in 0 2000 4000 8000 16000
			do
				for EXPR_NUM in {1..1}
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
					$cmd 2>&1 | tee $expr_dir/htap-$MODE-queryFresh-sf-$SF-ETL-$ETL-$QUERY-SEQ-COUNT-$SEQ_COUNT-$EXPR_NUM
					end=`date +%s`
					
					echo "Duration: $((($(date +%s)-$start)/60)) minutes"
					kill $(ps aux | grep htap | awk '{print $2}')

					echo "Sleeping for 3 seconds before next expr."
					sleep 3
				done
			done
	done

	#S3_NI
	for SF in 30
	do
		for ETL in 0 2000 4000 8000 16000
			do
				for EXPR_NUM in {1..1}
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
					$cmd 2>&1 | tee $expr_dir/htap-$MODE-queryFresh-sf-$SF-ETL-$ETL-$QUERY-SEQ-COUNT-$SEQ_COUNT-$EXPR_NUM
					end=`date +%s`
					
					echo "Duration: $((($(date +%s)-$start)/60)) minutes"
					kill $(ps aux | grep htap | awk '{print $2}')

					echo "Sleeping for 3 seconds before next expr."
					sleep 3
				done
			done
	done


	#ADAPTIVE
	for SF in 30
	do
		for ADP_RATIO in 0.25 0.5 0.75 1
			do
			for EXPR_NUM in {1..1}
			do
				MODE="ADAPTIVE"			
				start=`date +%s`
				echo "#####################  ADAPTIVE  ###########################"
				echo "#######################################################"
				echo "PerQueryFreshness: True"
				echo "Experiment #: $EXPR_NUM"
				echo "#######################################################"
				echo "#######################################################"
				cmd="./htap-ch-bench --ch-scale-factor=$SF --htap-mode=$MODE --num-olap-repeat=$SEQ_COUNT --adaptive-ratio=$ADP_RATIO"
				echo "$cmd"
				$cmd 2>&1 | tee $expr_dir/htap-$MODE-RATIO-$ADP_RATIO-queryFresh-sf-$SF-$QUERY-SEQ-COUNT-$SEQ_COUNT-$EXPR_NUM
				end=`date +%s`
				
				echo "Duration: $((($(date +%s)-$start)/60)) minutes"
				kill $(ps aux | grep htap | awk '{print $2}')

				echo "Sleeping for 3 seconds before next expr."
				sleep 3
			done
		done	
	done

	#ADAPTIVE-NI
	for SF in 30
	do
		for ADP_RATIO in 0.25 0.5 0.75 1
			do
			for EXPR_NUM in {1..1}
			do
				MODE="ADAPTIVE"			
				start=`date +%s`
				echo "#####################  ADAPTIVE  ###########################"
				echo "#######################################################"
				echo "PerQueryFreshness: True"
				echo "Experiment #: $EXPR_NUM"
				echo "#######################################################"
				echo "#######################################################"
				cmd="./htap-ch-bench --ch-scale-factor=$SF --htap-mode=$MODE --num-olap-repeat=$SEQ_COUNT --oltp-elastic-threshold=$ELASTIC_THRESH --adaptive-ratio=$ADP_RATIO"
				echo "$cmd"
				$cmd 2>&1 | tee $expr_dir/htap-$MODE-RATIO-$ADP_RATIO-queryFresh-sf-$SF-$QUERY-SEQ-COUNT-$SEQ_COUNT-$EXPR_NUM
				end=`date +%s`
				
				echo "Duration: $((($(date +%s)-$start)/60)) minutes"
				kill $(ps aux | grep htap | awk '{print $2}')

				echo "Sleeping for 3 seconds before next expr."
				sleep 3
			done
		done	
	done
done 


#-------------
# for SEQ_COUNT in 110
# do
# 	#ISOLATED
# 	for SF in 30
# 	do
# 			for EXPR_NUM in {1..1}
# 			do
# 				MODE="ISOLATED"			
# 				start=`date +%s`
# 				echo "#####################  ISOLATED  ###########################"
# 				echo "#######################################################"
# 				echo "PerQueryFreshness: False"
# 				echo "Experiment #: $EXPR_NUM"
# 				echo "#######################################################"
# 				echo "#######################################################"
# 				cmd="./htap-ch-bench --ch-scale-factor=$SF --htap-mode=$MODE --num-olap-repeat=$SEQ_COUNT --per-query-snapshot=false"
# 				echo "$cmd"
# 				$cmd 2>&1 | tee $expr_dir/htap-$MODE-sf-$SF-$QUERY-SEQ-COUNT-$SEQ_COUNT-$EXPR_NUM
# 				end=`date +%s`
				
# 				echo "Duration: $((($(date +%s)-$start)/60)) minutes"
# 				kill $(ps aux | grep htap | awk '{print $2}')

# 				echo "Sleeping for 3 seconds before next expr."
# 				sleep 3
# 			done
			
# 	done


# 	#COLOCATED
# 	for SF in 30
# 	do
# 			for EXPR_NUM in {1..1}
# 			do
# 				MODE="COLOCATED"			
# 				start=`date +%s`
# 				echo "#####################  COLOCATED  ###########################"
# 				echo "#######################################################"
# 				echo "PerQueryFreshness: False"
# 				echo "Experiment #: $EXPR_NUM"
# 				echo "#######################################################"
# 				echo "#######################################################"
# 				cmd="./htap-ch-bench --ch-scale-factor=$SF --htap-mode=$MODE --num-olap-repeat=$SEQ_COUNT --per-query-snapshot=false"
# 				echo "$cmd"
# 				$cmd 2>&1 | tee $expr_dir/htap-$MODE-sf-$SF-$QUERY-SEQ-COUNT-$SEQ_COUNT-$EXPR_NUM
# 				end=`date +%s`
				
# 				echo "Duration: $((($(date +%s)-$start)/60)) minutes"
# 				kill $(ps aux | grep htap | awk '{print $2}')

# 				echo "Sleeping for 3 seconds before next expr."
# 				sleep 3
# 			done
			
# 	done

# 	#S3_IS
# 	for SF in 30
# 	do
# 			for ETL in 0 2000 4000 8000 16000
# 			do
# 				for EXPR_NUM in {1..1}
# 				do
# 					MODE="HYBRID-ISOLATED"			
# 					start=`date +%s`
# 					echo "#####################  HYBRID-ISOLATED  ###########################"
# 					echo "#######################################################"
# 					echo "PerQueryFreshness: False"
# 					echo "Experiment #: $EXPR_NUM"
# 					echo "#######################################################"
# 					echo "#######################################################"
# 					cmd="./htap-ch-bench --ch-scale-factor=$SF --htap-mode=$MODE --num-olap-repeat=$SEQ_COUNT --etl-interval-ms=$ETL --per-query-snapshot=false"
# 					echo "$cmd"
# 					$cmd 2>&1 | tee $expr_dir/htap-$MODE-sf-$SF-ETL-$ETL-$QUERY-SEQ-COUNT-$SEQ_COUNT-$EXPR_NUM
# 					end=`date +%s`
					
# 					echo "Duration: $((($(date +%s)-$start)/60)) minutes"
# 					kill $(ps aux | grep htap | awk '{print $2}')

# 					echo "Sleeping for 3 seconds before next expr."
# 					sleep 3
# 				done
# 			done
# 	done

# 	#S3_NI
# 	for SF in 30
# 	do
# 		for ETL in 0 2000 4000 8000 16000
# 			do
# 				for EXPR_NUM in {1..1}
# 				do
# 					MODE="HYBRID-COLOC"			
# 					start=`date +%s`
# 					echo "#####################  HYBRID-COLOC  ###########################"
# 					echo "#######################################################"
# 					echo "PerQueryFreshness: False"
# 					echo "Experiment #: $EXPR_NUM"
# 					echo "#######################################################"
# 					echo "#######################################################"
# 					cmd="./htap-ch-bench --ch-scale-factor=$SF --htap-mode=$MODE --num-olap-repeat=$SEQ_COUNT --oltp-elastic-threshold=$ELASTIC_THRESH  --etl-interval-ms=$ETL --per-query-snapshot=false"
# 					echo "$cmd"
# 					$cmd 2>&1 | tee $expr_dir/htap-$MODE-sf-$SF-ETL-$ETL-$QUERY-SEQ-COUNT-$SEQ_COUNT-$EXPR_NUM
# 					end=`date +%s`
					
# 					echo "Duration: $((($(date +%s)-$start)/60)) minutes"
# 					kill $(ps aux | grep htap | awk '{print $2}')

# 					echo "Sleeping for 3 seconds before next expr."
# 					sleep 3
# 				done
# 			done
# 	done
# done

