select /*+ QUERY_INFO(name='SSB100::Q1.2') */ sum(lo_extendedprice*lo_discount) as revenue
from ssbm_lineorder, ssbm_date
where lo_orderdate = d_datekey
 and d_yearmonthnum = 199401
 and lo_discount between 4 and 6
 and lo_quantity between 26 and 35 ;
