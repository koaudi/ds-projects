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




select a.*, CASE WHEN prev_sales > 0 THEN (sales - prev_sales) / (prev_sales) END as percent_difference
FROM
(
select 
	country,
	month(timestamp),
	sum(saleValue * commissionPercent) as revenue
	LAG(sum(saleValue * commissionPercent)) OVER (ORDER BY country, month(timestamp) ) as prev_sales
	from
	transactions
	group by
	country, month(timestamp)) a;

	




	select 
	country,
	month,
	sum(saleValue * commissionPercent) as revenue
	LAG(sum(saleValue * commissionPercent)) OVER (ORDER BY country, month ) as prev_sales
	from
	transactions
	group by
	country, month;




(SELECT revenue FROM transactions prev WHERE prev.month(timestamp) < t.month(timestamp) ORDER BY  t.month(timestamp)DESC LIMIT 1) AS changes

select *,
(SELECT revenue FROM transactions prev WHERE prev.month(timestamp) < t.month(timestamp) ORDER BY  t.month(timestamp)DESC LIMIT 1) AS changes
FROM(
select 
t.country,
t.month(timestamp),
sum(saleValue * commissionPercent) as revenue
from
transactions t
group by
country, month)
group by country,month;


SELECT t.*,
 amount - (SELECT amount FROM transactions prev WHERE prev.date_end     < t.date_start ORDER BY date_start DESC LIMIT 1) AS changes
 FROM   transactions t


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