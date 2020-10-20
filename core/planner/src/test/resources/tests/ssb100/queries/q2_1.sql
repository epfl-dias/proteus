select /*+ QUERY_INFO(name='SSB100::Q2.1') */ sum(lo_revenue), d_year, p_brand1
from ssbm_lineorder, ssbm_part, ssbm_supplier, ssbm_date
where lo_orderdate = d_datekey
 and lo_partkey = p_partkey
 and lo_suppkey = s_suppkey
 and p_category = 'MFGR#12'
 and s_region = 'AMERICA'
group by d_year, p_brand1
order by d_year, p_brand1 ;
