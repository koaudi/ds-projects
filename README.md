# ds-learnings
git clone git clone https://github.com/koaudi/ds-learnings

Below is an outline for creating dynamic prize or pricing info 
SMS Message 'Buy at Kasha and stand a change to win 10,000 KSH or one of many prize'
When order is completed do the following steps
1. Check if customer han't wont before
2. Check available stock for prize
3. Check customer life time value 
4. Pick prize with closest possible value to clt
5. Check goal attainment for the day (if low increase chances of winning prize or reduce if high)
6. Randomly assign customer into win or lose depending on time of day, goal attainment, 
7. Add customer to prize table 

Use Cubes to have fast data retrieval
1. Order Id is fact table
2. All other tables are 2 column dimension tables indexed to order_id