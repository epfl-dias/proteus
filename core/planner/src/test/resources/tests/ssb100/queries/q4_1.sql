select /*+ QUERY_INFO(name='SSB100::Q4.1') */ d_year, c_nation, sum(lo_revenue - lo_supplycost) as profit
from ssbm_date, ssbm_customer, ssbm_supplier, ssbm_part, ssbm_lineorder
where lo_custkey = c_custkey
 and lo_suppkey = s_suppkey
 and lo_partkey = p_partkey
 and lo_orderdate = d_datekey
 and c_region = 'AMERICA'
 and s_region = 'AMERICA'
 and (p_mfgr = 'MFGR#1' or p_mfgr = 'MFGR#2')
group by d_year, c_nation
order by d_year, c_nation ;
