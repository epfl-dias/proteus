#include "util/radix/aggregations/radix-aggr.hpp"

/* #define RADIX_HASH(V)  ((V>>7)^(V>>13)^(V>>21)^V) */
#define HASH_BIT_MODULO(K, MASK, NBITS) (((K) & MASK) >> NBITS)

#ifndef NEXT_POW_2
/**
 *  compute the next number, greater than or equal to 32-bit unsigned v.
 *  taken from "bit twiddling hacks":
 *  http://graphics.stanford.edu/~seander/bithacks.html
 */
#define NEXT_POW_2(V)                           \
    do {                                        \
        V--;                                    \
        V |= V >> 1;                            \
        V |= V >> 2;                            \
        V |= V >> 4;                            \
        V |= V >> 8;                            \
        V |= V >> 16;                           \
        V++;                                    \
    } while(0)
#endif

#define MAX(X, Y) (((X) > (Y)) ? (X) : (Y))

#ifdef SYNCSTATS
#define SYNC_TIMERS_START(A, TID)               \
    do {                                        \
        uint64_t tnow;                          \
        startTimer(&tnow);                      \
        A->localtimer.sync1[0]      = tnow;     \
        A->localtimer.sync1[1]      = tnow;     \
        A->localtimer.sync3         = tnow;     \
        A->localtimer.sync4         = tnow;     \
        A->localtimer.finish_time   = tnow;     \
        if(TID == 0) {                          \
            A->globaltimer->sync1[0]    = tnow; \
            A->globaltimer->sync1[1]    = tnow; \
            A->globaltimer->sync3       = tnow; \
            A->globaltimer->sync4       = tnow; \
            A->globaltimer->finish_time = tnow; \
        }                                       \
    } while(0)

#define SYNC_TIMER_STOP(T) stopTimer(T)
#define SYNC_GLOBAL_STOP(T, TID) if(TID==0){ stopTimer(T); }
#else
#define SYNC_TIMERS_START(A, TID)
#define SYNC_TIMER_STOP(T)
#define SYNC_GLOBAL_STOP(T, TID)
#endif

/** Debug msg logging method */
#ifdef DEBUG
#define DEBUGMSG(COND, MSG, ...)                                    \
    if(COND) { fprintf(stdout, "[DEBUG] " MSG, ## __VA_ARGS__); }
#else
#define DEBUGMSG(COND, MSG, ...)
#endif

/* just to enable compilation with g++ */
#if defined(__cplusplus)
#define restrict __restrict__
#endif

/** checks malloc() result */
#ifndef MALLOC_CHECK
#define MALLOC_CHECK(M)                                                 \
    if(!M){                                                             \
        printf("[ERROR] MALLOC_CHECK: %s : %d\n", __FILE__, __LINE__);  \
        perror(": malloc() failed!\n");                                 \
        exit(EXIT_FAILURE);                                             \
    }
#endif

/**
 * Radix clustering algorithm which does not put padding in between
 * clusters.
 *
 * @param outRel
 * @param inRel
 * @param hist
 * @param R
 * @param D
 */
void
radix_cluster_nopadding(agg::relation_t * outRel, agg::relation_t * inRel, int R, int D)
{
	agg::tuple_t ** dst;
	agg::tuple_t * input;
    /* tuple_t ** dst_end; */
    uint32_t * tuples_per_cluster;
    uint32_t i;
    uint32_t offset;
    const uint32_t M = ((1 << D) - 1) << R;
    const uint32_t fanOut = 1 << D;
    const uint32_t ntuples = inRel->num_tuples;

    tuples_per_cluster = (uint32_t*)calloc(fanOut, sizeof(uint32_t));
    /* the following are fixed size when D is same for all the passes,
       and can be re-used from call to call. Allocating in this function
       just in case D differs from call to call. */
    dst     = (agg::tuple_t**)malloc(sizeof(agg::tuple_t*)*fanOut);
    /* dst_end = (tuple_t**)malloc(sizeof(tuple_t*)*fanOut); */

    input = inRel->tuples;
    /* count tuples per cluster */
    for( i=0; i < ntuples; i++ ){
        uint32_t idx = (uint32_t)(HASH_BIT_MODULO(input->key, M, R));
        tuples_per_cluster[idx]++;
        input++;
    }

    offset = 0;
    /* determine the start and end of each cluster depending on the counts. */
    for ( i=0; i < fanOut; i++ ) {
        dst[i]      = outRel->tuples + offset;
        offset     += tuples_per_cluster[i];
        /* dst_end[i]  = outRel->tuples + offset; */
    }

    input = inRel->tuples;
    /* copy tuples to their corresponding clusters at appropriate offsets */
    for( i=0; i < ntuples; i++ ){
        uint32_t idx   = (uint32_t)(HASH_BIT_MODULO(input->key, M, R));
        *dst[idx] = *input;
        ++dst[idx];
        input++;
        /* we pre-compute the start and end of each cluster, so the following
           check is unnecessary */
        /* if(++dst[idx] >= dst_end[idx]) */
        /*     REALLOCATE(dst[idx], dst_end[idx]); */
    }

    /* clean up temp */
    /* free(dst_end); */
    free(dst);
    free(tuples_per_cluster);
}

void
radix_cluster_nopadding(agg::tuple_t * outTuples, agg::tuple_t * inTuples, size_t num_tuples, int R, int D)
{
	agg::tuple_t ** dst;
	agg::tuple_t * input;
    /* tuple_t ** dst_end; */
    uint32_t * tuples_per_cluster;
    uint32_t i;
    uint32_t offset;
    const uint32_t M = ((1 << D) - 1) << R;
    const uint32_t fanOut = 1 << D;
    const uint32_t ntuples = num_tuples;

    tuples_per_cluster = (uint32_t*)calloc(fanOut, sizeof(uint32_t));
    /* the following are fixed size when D is same for all the passes,
       and can be re-used from call to call. Allocating in this function
       just in case D differs from call to call. */
    dst     = (agg::tuple_t**)malloc(sizeof(agg::tuple_t*)*fanOut);
    /* dst_end = (tuple_t**)malloc(sizeof(tuple_t*)*fanOut); */

    input = inTuples;
    /* count tuples per cluster */
    for( i=0; i < ntuples; i++ ){
    	uint32_t idx = (uint32_t)(HASH_BIT_MODULO(input->key, M, R));
    	tuples_per_cluster[idx]++;
        input++;
    }

    offset = 0;
    /* determine the start and end of each cluster depending on the counts. */
    for ( i=0; i < fanOut; i++ ) {
        dst[i]      = outTuples + offset;
        offset     += tuples_per_cluster[i];
        /* dst_end[i]  = outRel->tuples + offset; */
    }

    input = inTuples;
    // int cnt = 0;
    /* copy tuples to their corresponding clusters at appropriate offsets */
    for( i=0; i < ntuples; i++ ){
    	uint32_t idx   = (uint32_t)(HASH_BIT_MODULO(input->key, M, R));
//    	cout << "[radix_cluster_nopadding: ] cluster: "<< idx <<" key? " << input->key << endl;
//        if(R!=0)	{
//        	cout << input->key << endl;
//        	cnt++;
//        }
        *dst[idx] = *input;
        ++dst[idx];
        input++;
        /* we pre-compute the start and end of each cluster, so the following
           check is unnecessary */
        /* if(++dst[idx] >= dst_end[idx]) */
        /*     REALLOCATE(dst[idx], dst_end[idx]); */

    }
//    if(R!=0)
//    	cout << "How many tuples? " << cnt << endl;
    /* clean up temp */
    /* free(dst_end); */
    free(dst);
    free(tuples_per_cluster);
}

/**
 *  This algorithm builds the hashtable using the bucket chaining idea and used
 *  in PRO implementation. Join between given two relations is evaluated using
 *  the "bucket chaining" algorithm proposed by Manegold et al. It is used after
 *  the partitioning phase, which is common for all algorithms. Moreover, R and
 *  S typically fit into L2 or at least R and |R|*sizeof(int) fits into L2 cache.
 *
 * @param R input relation R
 * @param S input relation S
 *
 * @return number of result tuples
 */

/**
 * CHANGELOG
 * Used in the context of RJStepwise
 * (when S side is partitioned)
 */

void bucket_chaining_agg_prepare(const agg::relation_t * const R, HT * ht)	{
    const uint32_t numR = R->num_tuples;
    uint32_t N = numR;

    NEXT_POW_2(N);
    /* N <<= 1; */
    const uint32_t MASK = (N-1) << (NUM_RADIX_BITS);
    ht->mask = MASK;
    ht->next   = (int*) malloc(sizeof(int) * numR);
    ht->bucket = (int*) calloc(N, sizeof(int));

    for(uint32_t i=0; i < numR; ){
        uint32_t idx = HASH_BIT_MODULO(R->tuples[i].key, MASK, NUM_RADIX_BITS);
        (ht->next)[i]      = (ht->bucket)[idx];
        (ht->bucket)[idx]  = ++i;     /* we start pos's from 1 instead of 0 */

    }
}

void bucket_chaining_agg_prepare(const agg::tuple_t * const tuplesR, int num_tuples, HT * ht)	{
    const uint32_t numR = num_tuples;
    uint32_t N = numR;

    NEXT_POW_2(N);
    /* N <<= 1; */
    const uint32_t MASK = (N-1) << (NUM_RADIX_BITS);
    ht->mask = MASK;
    ht->next   = (int*) malloc(sizeof(int) * numR);
    ht->bucket = (int*) calloc(N, sizeof(int));
    //cout << "[PREPARING]: " << endl;
    for(uint32_t i=0; i < numR; ){
        uint32_t idx = HASH_BIT_MODULO(tuplesR[i].key, MASK, NUM_RADIX_BITS);
        //cout << "[K]: " << tuplesR[i].key << " [V]: " << tuplesR[i].payload << endl;
        (ht->next)[i]      = (ht->bucket)[idx];
        (ht->bucket)[idx]  = ++i;     /* we start pos's from 1 instead of 0 */
    }
}

/**
 * Used in the context of RJStepwise
 * (when S side is partitioned)
 */
int64_t
bucket_chaining_agg_probe(const agg::relation_t * const R, HT * ht,
                     const agg::tuple_t * const s)
{
	int64_t matches = 0;
    uint32_t idx = HASH_BIT_MODULO(s->key, ht->mask, NUM_RADIX_BITS);
    const agg::tuple_t * const Rtuples = R->tuples;

    for(int hit = (ht->bucket)[idx]; hit > 0; hit = (ht->next)[hit-1]){
    	if (s->key == Rtuples[hit - 1].key)
		{
			matches++;
		}
    }
    /* PROBE-LOOP END  */
    return matches;
}

/**
 * Used in the context of RJStepwise
 * (when S side is partitioned)
 */
void bucket_chaining_agg_finish(HT * ht)
{
	/* clean up temp */
	free(ht->bucket);
	free(ht->next);
}

/**
 * @param num_tuples size of input relation
 * @param inTuples	 ht entries of relation - not actual data
 *
 * @return item count per cluster defined
 */
int *partitionHT(size_t num_tuples, agg::tuple_t *inTuples)	{
	size_t sz = num_tuples * sizeof(agg::tuple_t) + RELATION_PADDING_AGG;
	agg::tuple_t* outTuples = (agg::tuple_t*) malloc(sz);

	/* apply radix-clustering on relation for pass-1 */
	radix_cluster_nopadding(outTuples, inTuples, num_tuples, 0, NUM_RADIX_BITS/NUM_PASSES);

	/* apply radix-clustering for pass-2 */
	radix_cluster_nopadding(inTuples, outTuples, num_tuples,
	                            NUM_RADIX_BITS/NUM_PASSES,
	                            NUM_RADIX_BITS-(NUM_RADIX_BITS/NUM_PASSES));
	free(outTuples);

	int * count_per_cluster = (int*) calloc((1 << NUM_RADIX_BITS), sizeof(int));

	/* compute number of tuples per cluster */
	for (size_t i = 0; i < num_tuples; i++) {
		size_t idx = (inTuples[i].key) & ((1 << NUM_RADIX_BITS) - 1);
		count_per_cluster[idx]++;
	}
	return count_per_cluster;
}

int64_t
RJStepwise(agg::relation_t * relR, agg::relation_t * relS)
{
    int64_t result = 0;
    uint32_t i;

    agg::relation_t *outRelR, *outRelS;

    outRelR = (agg::relation_t*) malloc(sizeof(agg::relation_t));
    outRelS = (agg::relation_t*) malloc(sizeof(agg::relation_t));

    /* allocate temporary space for partitioning */
    size_t sz = relR->num_tuples * sizeof(agg::tuple_t) + RELATION_PADDING_AGG;
    outRelR->tuples     = (agg::tuple_t*) malloc(sz);
    outRelR->num_tuples = relR->num_tuples;

    sz = relS->num_tuples * sizeof(agg::tuple_t) + RELATION_PADDING_AGG;
    outRelS->tuples     = (agg::tuple_t*) malloc(sz);
    outRelS->num_tuples = relS->num_tuples;

    /***** do the multi-pass (2) partitioning *****/

    /* apply radix-clustering on relation R for pass-1 */
    radix_cluster_nopadding(outRelR, relR, 0, NUM_RADIX_BITS/NUM_PASSES);

    /* apply radix-clustering on relation S for pass-1 */
    radix_cluster_nopadding(outRelS, relS, 0, NUM_RADIX_BITS/NUM_PASSES);

    /* apply radix-clustering on relation R for pass-2 */
    radix_cluster_nopadding(relR, outRelR,
                            NUM_RADIX_BITS/NUM_PASSES,
                            NUM_RADIX_BITS-(NUM_RADIX_BITS/NUM_PASSES));

    /* apply radix-clustering on relation S for pass-2 */
    radix_cluster_nopadding(relS, outRelS,
                            NUM_RADIX_BITS/NUM_PASSES,
                            NUM_RADIX_BITS-(NUM_RADIX_BITS/NUM_PASSES));

    /* clean up temporary relations */
    free(outRelR->tuples);
    free(outRelS->tuples);
    free(outRelR);
    free(outRelS);

    int * R_count_per_cluster = (int*)calloc((1<<NUM_RADIX_BITS), sizeof(int));
    int * S_count_per_cluster = (int*)calloc((1<<NUM_RADIX_BITS), sizeof(int));

    /* compute number of tuples per cluster */
    for( i=0; i < relR->num_tuples; i++ ){
        uint32_t idx = (relR->tuples[i].key) & ((1<<NUM_RADIX_BITS)-1);
        R_count_per_cluster[idx] ++;
    }
    for( i=0; i < relS->num_tuples; i++ ){
        uint32_t idx = (relS->tuples[i].key) & ((1<<NUM_RADIX_BITS)-1);
        S_count_per_cluster[idx] ++;
    }

    /*
     * Dependencies from code above:
     * -> Counts (can add it to HT structure)
     * -> Materialized Relations
     */

    /* build hashtable on inner */
    int r, s; /* start index of next clusters */
    r = s = 0;

    /* Memory Manager: Take care of HT* demands */
    HT *HT_per_cluster = (HT*)calloc((1<<NUM_RADIX_BITS), sizeof(HT));
	for (i = 0; i < (1 << NUM_RADIX_BITS); i++)
	{
		agg::relation_t tmpR, tmpS;

		if (R_count_per_cluster[i] > 0 && S_count_per_cluster[i] > 0)
		{

			tmpR.num_tuples = R_count_per_cluster[i];
			tmpR.tuples = relR->tuples + r;
			r += R_count_per_cluster[i];

			tmpS.num_tuples = S_count_per_cluster[i];
			tmpS.tuples = relS->tuples + s;
			s += S_count_per_cluster[i];

			/* Memory Manager: Take care of HT* demands */
			bucket_chaining_agg_prepare(&tmpR, &(HT_per_cluster[i]));

			for (size_t j = 0; j < tmpS.num_tuples; j++)
			{
				result += bucket_chaining_agg_probe(&tmpR,
						&(HT_per_cluster[i]), &(tmpS.tuples[j]));
			}

			bucket_chaining_agg_finish(&(HT_per_cluster[i]));
		}
		else
		{
			r += R_count_per_cluster[i];
			s += S_count_per_cluster[i];
		}
	}

    /* clean-up temporary buffers */
    free(S_count_per_cluster);
    free(R_count_per_cluster);

    return result;
}
