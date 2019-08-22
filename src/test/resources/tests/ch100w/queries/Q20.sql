select	 su_name, su_address
from	 ch100w_supplier, ch100w_nation
where	 su_suppkey in
		(select  mod(s_i_id * s_w_id, 10000)
		from     ch100w_stock, ch100w_orderline
		where    s_i_id in
				(select i_id
				 from ch100w_item
				 where i_data like 'co%')
			 and ol_i_id=s_i_id
			 and ol_delivery_d > timestamp '2010-05-23 12:00:00'
		group by s_i_id, s_w_id, s_quantity
		having   2*s_quantity > sum(ol_quantity))
	 and su_nationkey = n_nationkey
	 and n_name = 'Germany'
order by su_name ;