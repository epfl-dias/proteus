-- CH-Benchmark (Analytical Queries)

-- Q1

select   ol_number,
     sum(ol_quantity) as sum_qty,
     sum(ol_amount) as sum_amount,
     avg(ol_quantity) as avg_qty,
     avg(ol_amount) as avg_amount,
     count(*) as count_order
from     ch100w_orderline
where    ol_delivery_d > '2007-01-02 00:00:00.000000'
group by ol_number order by 


-- Q2

select   s_suppkey, s_name, n_name, i_id, i_name, s_address, s_phone, s_comment
from     ch100w_item, ch100w_supplier, ch100w_stock, ch100w_nation, ch100w_region,
     (select s_i_id as m_i_id,
         min(s_quantity) as m_s_quantity
     from    ch100w_stock, ch100w_supplier, ch100w_nation, ch100w_region
     where   mod((s_w_id*s_i_id),10000)=s_suppkey
         and s_nationkey=n_nationkey
         and n_regionkey=r_regionkey
         and r_name like 'Europ%'
     group by s_i_id) m
where    i_id = s_i_id
     and mod((s_w_id * s_i_id), 10000) = s_suppkey
     and s_nationkey = n_nationkey
     and n_regionkey = r_regionkey
     and i_data like '%b'
     and r_name like 'Europ%'
     and i_id=m_i_id
     and s_quantity = m_s_quantity
order by n_name, s_name, i_id


-- Q3

select   ol_o_id, ol_w_id, ol_d_id,
     sum(ol_amount) as revenue, o_entry_d
from     ch100w_customer, ch100w_neworder, ch100w_order, ch100w_orderline
where    c_state like 'A%'
     and c_id = o_c_id
     and c_w_id = o_w_id
     and c_d_id = o_d_id
     and no_w_id = o_w_id
     and no_d_id = o_d_id
     and no_o_id = o_id
     and ol_w_id = o_w_id
     and ol_d_id = o_d_id
     and ol_o_id = o_id
     and o_entry_d > '2007-01-02 00:00:00.000000'
group by ol_o_id, ol_w_id, ol_d_id, o_entry_d
order by revenue desc, o_entry_d


-- Q4

select  o_ol_cnt, count(o_id) as order_count
from    ch100w_order
where   o_entry_d >= '2007-01-02'
    and o_entry_d < '2012-01-02'
    and exists (select *
            from ch100w_orderline
            where o_id = ol_o_id
            and o_w_id = ol_w_id
            and o_d_id = ol_d_id
            and ol_delivery_d >= o_entry_d)
group   by o_ol_cnt
order   by o_ol_cnt

-- Q5

select   n_name,
     sum(ol_amount) as revenue
from     ch100w_customer, ch100w_order, ch100w_orderline, ch100w_stock, ch100w_supplier, ch100w_nation, ch100w_region
where    c_id = o_c_id
     and c_w_id = o_w_id
     and c_d_id = o_d_id
     and ol_o_id = o_id
     and ol_w_id = o_w_id
     and ol_d_id=o_d_id
     and ol_w_id = s_w_id
     and ol_i_id = s_i_id
     and mod((s_w_id * s_i_id),10000) = su_suppkey
     and ascii(substr(c_state,1,1)) = su_nationkey
     and s_nationkey = n_nationkey
     and n_regionkey = r_regionkey
     and r_name = 'Europe'
     and o_entry_d >= '2007-01-02 00:00:00.000000'
group by n_name
order by revenue desc


-- Q6

select  sum(ol_amount) as revenue
from    ch100w_orderline
where   ol_delivery_d >= '1999-01-01'
    and ol_delivery_d < '2020-01-01'
    and ol_quantity between 1 and 100000


-- Q7

select   su_nationkey as supp_nation,
     substr(c_state,1,1) as cust_nation,
     extract(year from o_entry_d) as l_year,
     sum(ol_amount) as revenue
from     supplier, stock, orderline, orders, customer, nation n1, nation n2
where    ol_supply_w_id = s_w_id
     and ol_i_id = s_i_id
     and mod((s_w_id * s_i_id), 10000) = su_suppkey
     and ol_w_id = o_w_id
     and ol_d_id = o_d_id
     and ol_o_id = o_id
     and c_id = o_c_id
     and c_w_id = o_w_id
     and c_d_id = o_d_id
     and su_nationkey = n1.n_nationkey
     and ascii(substr(c_state,1,1)) = n2.n_nationkey
     and (
        (n1.n_name = 'Germany' and n2.n_name = 'Cambodia')
         or
        (n1.n_name = 'Cambodia' and n2.n_name = 'Germany')
         )
     and ol_delivery_d between '2007-01-02 00:00:00.000000' and '2012-01-02 00:00:00.000000'
group by su_nationkey, substr(c_state,1,1), extract(year from o_entry_d)
order by su_nationkey, cust_nation, l_year


-- Q8

select   extract(year from o_entry_d) as l_year,
     sum(case when n2.n_name = 'Germany' then ol_amount else 0 end) / sum(ol_amount) as mkt_share
from     item, supplier, stock, orderline, orders, customer, nation n1, nation n2, region
where    i_id = s_i_id
     and ol_i_id = s_i_id
     and ol_supply_w_id = s_w_id
     and mod((s_w_id * s_i_id),10000) = su_suppkey
     and ol_w_id = o_w_id
     and ol_d_id = o_d_id
     and ol_o_id = o_id
     and c_id = o_c_id
     and c_w_id = o_w_id
     and c_d_id = o_d_id
     and n1.n_nationkey = ascii(substr(c_state,1,1))
     and n1.n_regionkey = r_regionkey
     and ol_i_id < 1000
     and r_name = 'Europe'
     and su_nationkey = n2.n_nationkey
     and o_entry_d between '2007-01-02 00:00:00.000000' and '2012-01-02 00:00:00.000000'
     and i_data like '%b'
     and i_id = ol_i_id
group by extract(year from o_entry_d)
order by l_year


-- Q9

select   n_name, extract(year from o_entry_d) as l_year, sum(ol_amount) as sum_profit
from     item, stock, supplier, orderline, orders, nation
where    ol_i_id = s_i_id
     and ol_supply_w_id = s_w_id
     and mod((s_w_id * s_i_id), 10000) = su_suppkey
     and ol_w_id = o_w_id
     and ol_d_id = o_d_id
     and ol_o_id = o_id
     and ol_i_id = i_id
     and su_nationkey = n_nationkey
     and i_data like '%BB'
group by n_name, extract(year from o_entry_d)
order by n_name, l_year desc


-- Q10

select   c_id, c_last, sum(ol_amount) as revenue, c_city, c_phone, n_name
from     customer, orders, orderline, nation
where    c_id = o_c_id
     and c_w_id = o_w_id
     and c_d_id = o_d_id
     and ol_w_id = o_w_id
     and ol_d_id = o_d_id
     and ol_o_id = o_id
     and o_entry_d >= '2007-01-02 00:00:00.000000'
     and o_entry_d <= ol_delivery_d
     and n_nationkey = ascii(substr(c_state,1,1))
group by c_id, c_last, c_city, c_phone, n_name
order by revenue desc


-- Q11

select   s_i_id, sum(s_order_cnt) as ordercount
from     stock, supplier, nation
where    mod((s_w_id * s_i_id),10000) = su_suppkey
     and su_nationkey = n_nationkey
     and n_name = 'Germany'
group by s_i_id
having   sum(s_order_cnt) >
        (select sum(s_order_cnt) * .005
        from stock, supplier, nation
        where mod((s_w_id * s_i_id),10000) = su_suppkey
        and su_nationkey = n_nationkey
        and n_name = 'Germany')
order by ordercount desc


-- Q12

select   o_ol_cnt,
     sum(case when o_carrier_id = 1 or o_carrier_id = 2 then 1 else 0 end) as high_line_count,
     sum(case when o_carrier_id <> 1 and o_carrier_id <> 2 then 1 else 0 end) as low_line_count
from     orders, orderline
where    ol_w_id = o_w_id
     and ol_d_id = o_d_id
     and ol_o_id = o_id
     and o_entry_d <= ol_delivery_d
     and ol_delivery_d < '2020-01-01 00:00:00.000000'
group by o_ol_cnt
order by o_ol_cnt


-- Q13

select   c_count, count(*) as custdist
from     (select c_id, count(o_id)
     from customer left outer join orders on (
        c_w_id = o_w_id
        and c_d_id = o_d_id
        and c_id = o_c_id
        and o_carrier_id > 8)
     group by c_id) as c_orders (c_id, c_count)
group by c_count
order by custdist desc, c_count desc


-- Q14

select  100.00 * sum(case when i_data like 'PR%' then ol_amount else 0 end) / (1+sum(ol_amount)) as promo_revenue
from    orderline, item
where   ol_i_id = i_id and ol_delivery_d >= '2007-01-02 00:00:00.000000'
    and ol_delivery_d < '2020-01-02 00:00:00.000000'


-- Q15

with     revenue (supplier_no, total_revenue) as (
     select mod((s_w_id * s_i_id),10000) as supplier_no,
        sum(ol_amount) as total_revenue
     from   orderline, stock
        where ol_i_id = s_i_id and ol_supply_w_id = s_w_id
        and ol_delivery_d >= '2007-01-02 00:00:00.000000'
     group by mod((s_w_id * s_i_id),10000))
select   su_suppkey, su_name, su_address, su_phone, total_revenue
from     supplier, revenue
where    su_suppkey = supplier_no
     and total_revenue = (select max(total_revenue) from revenue)
order by su_suppkey


-- Q16

select   i_name,
     substr(i_data, 1, 3) as brand,
     i_price,
     count(distinct (mod((s_w_id * s_i_id),10000))) as supplier_cnt
from     stock, item
where    i_id = s_i_id
     and i_data not like 'zz%'
     and (mod((s_w_id * s_i_id),10000) not in
        (select su_suppkey
         from supplier
         where su_comment like '%bad%'))
group by i_name, substr(i_data, 1, 3), i_price
order by supplier_cnt desc


-- Q17

select  sum(ol_amount) / 2.0 as avg_yearly
from    orderline, (select   i_id, avg(ol_quantity) as a
            from     item, orderline
            where    i_data like '%b'
                 and ol_i_id = i_id
            group by i_id) t
where   ol_i_id = t.i_id
    and ol_quantity < t.a


-- Q18

select   c_last, c_id o_id, o_entry_d, o_ol_cnt, sum(ol_amount)
from     customer, orders, orderline
where    c_id = o_c_id
     and c_w_id = o_w_id
     and c_d_id = o_d_id
     and ol_w_id = o_w_id
     and ol_d_id = o_d_id
     and ol_o_id = o_id
group by o_id, o_w_id, o_d_id, c_id, c_last, o_entry_d, o_ol_cnt
having   sum(ol_amount) > 200
order by sum(ol_amount) desc, o_entry_d


-- Q19

select  sum(ol_amount) as revenue
from    orderline, item
where   (
      ol_i_id = i_id
          and i_data like '%a'
          and ol_quantity >= 1
          and ol_quantity <= 10
          and i_price between 1 and 400000
          and ol_w_id in (1,2,3)
    ) or (
      ol_i_id = i_id
      and i_data like '%b'
      and ol_quantity >= 1
      and ol_quantity <= 10
      and i_price between 1 and 400000
      and ol_w_id in (1,2,4)
    ) or (
      ol_i_id = i_id
      and i_data like '%c'
      and ol_quantity >= 1
      and ol_quantity <= 10
      and i_price between 1 and 400000
      and ol_w_id in (1,5,3)
    )


-- Q20

select   su_name, su_address
from     supplier, nation
where    su_suppkey in
        (select  mod(s_i_id * s_w_id, 10000)
        from     stock, orderline
        where    s_i_id in
                (select i_id
                 from item
                 where i_data like 'co%')
             and ol_i_id=s_i_id
             and ol_delivery_d > '2010-05-23 12:00:00'
        group by s_i_id, s_w_id, s_quantity
        having   2*s_quantity > sum(ol_quantity))
     and su_nationkey = n_nationkey
     and n_name = 'Germany'
order by su_name


-- Q21

select   su_name, count(*) as numwait
from     supplier, orderline l1, orders, stock, nation
where    ol_o_id = o_id
     and ol_w_id = o_w_id
     and ol_d_id = o_d_id
     and ol_w_id = s_w_id
     and ol_i_id = s_i_id
     and mod((s_w_id * s_i_id),10000) = su_suppkey
     and l1.ol_delivery_d > o_entry_d
     and not exists (select *
             from   orderline l2
             where  l2.ol_o_id = l1.ol_o_id
                and l2.ol_w_id = l1.ol_w_id
                and l2.ol_d_id = l1.ol_d_id
                and l2.ol_delivery_d > l1.ol_delivery_d)
     and su_nationkey = n_nationkey
     and n_name = 'Germany'
group by su_name
order by numwait desc, su_name


-- Q22

select   substr(c_state,1,1) as country,
     count(*) as numcust,
     sum(c_balance) as totacctbal
from     customer
where    substr(c_phone,1,1) in ('1','2','3','4','5','6','7')
     and c_balance > (select avg(c_BALANCE)
              from   customer
              where  c_balance > 0.00
                 and substr(c_phone,1,1) in ('1','2','3','4','5','6','7'))
     and not exists (select *
             from   orders
             where  o_c_id = c_id
                    and o_w_id = c_w_id
                    and o_d_id = c_d_id)
group by substr(c_state,1,1)
order by substr(c_state,1,1)
