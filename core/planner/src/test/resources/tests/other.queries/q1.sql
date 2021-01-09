select count(lo_custkey), lo_orderdate
from ssbm_lineorder
group by lo_orderdate;
