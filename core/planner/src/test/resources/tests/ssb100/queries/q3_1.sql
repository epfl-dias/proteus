select /*+ QUERY_INFO(name='SSB100::Q3.1') */ c_nation, s_nation, d_year, sum(lo_revenue) as lo_revenue
from ssbm_customer, ssbm_lineorder, ssbm_supplier, ssbm_date
where lo_custkey = c_custkey
 and lo_suppkey = s_suppkey
 and lo_orderdate = d_datekey
 and c_region = 'ASIA'
 and s_region = 'ASIA'
 and d_year >= 1992
 and d_year <= 1997
group by c_nation, s_nation, d_year
order by d_year asc, lo_revenue desc ;
