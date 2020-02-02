DROP TEMPORARY TABLE transactions;

CREATE TEMPORARY TABLE transactions (
    country VARCHAR(255),
    saleValue INT,
    commissionPercent DECIMAL(3,2),
    month VARCHAR(255)
);



INSERT INTO 
   transactions (country,saleValue,commissionPercent,month)
VALUES
     ('KENYA',100,0.25,'2019-01'),
	 ('KENYA',200,0.25,'2019-02'),
	 ('KENYA',300,0.25,'2019-03'),
	 ('KENYA',400,0.25,'2019-04'),
	 ('RWANDA',100,0.25,'2019-01'),
	 ('RWANDA',100,0.25,'2019-02'),
	 ('RWANDA',100,0.25,'2019-03'),
	 ('RWANDA',100,0.25,'2019-04'),
	 ('UGANDA',100,0.25,'2019-01'),
	 ('UGANDA',150,0.25,'2019-02'),
	 ('UGANDA',150,0.25,'2019-03'),
	 ('UGANDA',150,0.25,'2019-04');




select 
country,
month(timestamp),
sum(saleValue * commissionPercent) as revenue,
100 * (count(*) - lag(count(*), 1) over (order by month(timestamp))) / lag(count(*), 1) over (order by month(timestamp))) as percent_difference
from
transactions
group by
country, month, 1;
order by 1



##QUESTION 3
MAX(coupon_text) will select SPRINGSAVINGS since the coupon_text column is a string the max function will sort the rows alphabetically asceding and pick the last rows
















SELECT
  country,
  month,
  revenue,
  CAST(100 * sum(revenue) OVER (ORDER BY month) 
           / sum(revenue) OVER () AS numeric(10, 2)) percentage
FROM (
  select 
	country,
	month,
	sum(saleValue * commissionPercent) as revenue
	from
	transactions
	group by
	country, month;
) p
ORDER BY country, month;
















select d.*, (revenue - prev_revenue) / (prev_revenue) as percent_difference
from (select d.*
             country,
			 month,
             sum(saleValue * commissionPercent) as revenue,
             lag(sum(saleValue * commissionPercent)) over (partition by country order by month) as prev_revenue
      from transactions
	  group by
	  country, month
     ) d;




select date_trunc('month', timestamp) as date,
       count(*) as count,
       100 * (count(*) - lag(count(*), 1) over (order by timestamp)) / lag(count(*), 1) over (order by timestamp)) || '%' as growth
from events
where event_name = 'created chart'
group by 1
order by 1