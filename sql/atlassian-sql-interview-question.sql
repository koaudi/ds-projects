/*Tables:

all_page_view

URL, user_id, page_name, page_key, date, timestamp
str, int, str, int, str, date, timestamp


all_clicks

URL, user_id, page_key, action_key, date, timestamp
str, int, int, str, date, timestamp



Question: What’s the distribution of the number of pages users engage with? (engagement defined by clicks on the pages in this case)
 
 X-axis: # of pages engaged with
 Y-axis: # of users engaged 
 
# of Pages  Users  
-----      -----
 1         10    
 2         20     
 3         30
 4         15
 5         3
 ...       ...
 */
 select LEFT(b.date,7), sum(pages), count(distinct user_id) as users from 
 (SELECT b.date, a.user_id, count(distinct page_key) as pages
 FROM all_page_view  a left join all_clicks b on a.user_id = b.user_id where action_key = 'click' and a.page_key = b.page_key
 Group By b.date,user_id)
 WHERE b.date between daterange1 and daterange2
 group by LEFT(b.date,7),pages,b.user_id; 
 
 /* 2020-01-30
 
 