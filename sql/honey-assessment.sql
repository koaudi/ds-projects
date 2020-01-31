select 
country,
month(timestamp) as month,
sum(saleValue * commissionPercent) as revenue
from
transactions
group by
country, month