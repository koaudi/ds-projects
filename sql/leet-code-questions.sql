LEET CODE QUESTIONS

#Get Nth highest salary
CREATE FUNCTION getNthHighestSalary(N INT) RETURNS INT
BEGIN
declare M INT;
SET M = N-1;
  RETURN (
      # Write your MySQL query statement below.
      select distinct Salary from Employee ORDER BY salary DESC limit 1 offset M
  );
END


#hacker rank
*/SELECT CONCAT(NAME,CONCAT('(',CONCAT(SUBSTR(OCCUPATION,1,1),')'))) FROM OCCUPATIONS ORDER BY NAME ASC;

SELECT 'There are a total of ' || COUNT(OCCUPATION) || ' ' || LOWER(OCCUPATION) || 's' FROM OCCUPATIONS GROUP BY OCCUPATION ORDER BY COUNT(OCCUPATION) ASC, OCCUPATION ASC;


### BINARY TREE 
SELECT N, CASE WHEN P IS NULL THEN 'Root' WHEN N IN (SELECT P FROM BST) THEN 'Inner' ELSE 'Leaf' END FROM BST ORDER BY N;

/*
Aggegregating multiple tables
*/
SELECT  c.company_code,
        c.founder,
       count(distinct e.lead_manager_code),
       count(distinct e.senior_manager_code),
       count(distinct e.manager_code),
       count(distinct e.employee_code)
FROM company c left join employee e
                on c.company_code = e.company_code
GROUP BY c.company_code, c.founder
ORDER BY SUBSTR(c.company_code,2,length(c.company_code));

#Manhattan distance
select round(abs(min(lat_n) - max(lat_n) + min(long_w) - max(long_w)),4) from station;

#Euclidian distance
select round(sqrt((min(lat_n) - max(lat_n))*(min(lat_n) - max(lat_n)) + ((min(long_w) - max(long_w))*(min(long_w) - max(long_w)))),4) from station;


#Median
select round(sqrt((min(lat_n) - max(lat_n))*(min(lat_n) - max(lat_n)) + ((min(long_w) - max(long_w))*(min(long_w) - max(long_w)))),4) from station;

#select grades between a range
select case when grade < 8 then null else name end,g.grade,s.marks from students s, grades g where
s.marks between g.min_mark and g.max_mark order by grade desc,s.name;	

### CREATING LEADER BOARD COUNT WHO HAS MORE THAN 1 HIGH SCORE
SELECT 
    h.hacker_id, h.name
FROM
    submissions s
        LEFT JOIN
    hackers h ON s.hacker_id = h.hacker_id
        LEFT JOIN
    challenges c ON s.challenge_id = c.challenge_id
        LEFT JOIN
    difficulty d ON d.difficulty_level = c.difficulty_level
WHERE
    s.score = d.score
        AND d.difficulty_level = c.difficulty_level
GROUP BY h.hacker_id , h.name
HAVING COUNT(h.hacker_id) > 1
ORDER BY COUNT(h.hacker_id) DESC , h.hacker_id ASC;


###A SUBQUERY TO GET MINIMUm
/*
Enter your query here.
*/
SELECT 
    W.ID, WP.AGE, W.COINS_NEEDED, W.POWER
FROM
    WANDS W
        JOIN
    WANDS_PROPERTY WP ON W.CODE = WP.CODE
WHERE
    WP.IS_EVIL = 0
        AND W.COINS_NEEDED = (SELECT 
            MIN(coins_needed)
        FROM
            Wands w1
                JOIN
            WANDS_PROPERTY p1 ON (w1.code = p1.code)
        WHERE
            w1.power = w.power AND p1.age = WP.age)
ORDER BY W.POWER DESC , WP.AGE DESC;


#over partition by does not reduce rows just adds aggregate
SELECT  d o
    first_name,
    last_name,
    department_id, 
    ROUND(AVG(salary) OVER (
        PARTITION BY department_id
    )) avg_department_salary
FROM
    employees;

##Suppose that a website contains two tables, the Customers table and the Orders table. Write a SQL query to find all customers who never order anything.
select Name as Customers from Customers c left join Orders o on c.Id = o.CustomerId where o.CustomerId is null

#Ranking scores
https://stackoverflow.com/questions/48837762/rank-scores-leetcode-178
select scores.Score, count(ranking.Score) as Rank
from scores, (select distinct Score from scores) ranking
where ranking.score>=scores.Score
group by scores.Id
order by scores.Score desc

## Question 197 Given a Weather table, write a SQL query to find all dates' Ids with higher temperature compared to its previous (yesterday's) dates.	
SELECT
    weather.id AS 'Id'
FROM
    weather
        JOIN
    weather w ON DATEDIFF(weather.date, w.date) = 1
        AND weather.Temperature > w.Temperature
;

## MODULAR
select * from cinema where mod(id,2) > 0 and description not like '%boring%' order by rating desc;

### 180 Consecutive numbers	
select distinct l1.Num as ConsecutiveNums from 
Logs l1, logs l2, logs l3 
where
l1.Id = l2.id - 1 
and l2.id = l3.id - 1
and l1.Num = l2.Num and l2.Num = l3.Num;

#MAX EARNINGS HACKER RANK
select salary*months, count(*) from employee
where salary*months = (select max(salary*months) from employee)
group by salary*months;


#HACKER RANK challenges problem - https://www.hackerrank.com/challenges/challenges/problem
/* these are the columns we want to output */
select c.hacker_id, h.name ,count(c.hacker_id) as c_count

/* this is the join we want to output them from */
from Hackers as h
    inner join Challenges as c on c.hacker_id = h.hacker_id

/* after they have been grouped by hacker */
group by c.hacker_id

/* but we want to be selective about which hackers we output */
/* having is required (instead of where) for filtering on groups */
having 

    /* output anyone with a count that is equal to... */
    c_count = 
        /* the max count that anyone has */
        (SELECT MAX(temp1.cnt)
        from (SELECT COUNT(hacker_id) as cnt
             from Challenges
             group by hacker_id
             order by hacker_id) temp1)

    /* or anyone who's count is in... */
    or c_count in 
        /* the set of counts... */
        (select t.cnt
         from (select count(*) as cnt 
               from challenges
               group by hacker_id) t
         /* who's group of counts... */
         group by t.cnt
         /* has only one element */
         having count(t.cnt) = 1)

/* finally, the order the rows should be output */
order by c_count DESC, c.hacker_id

/* ;) */
;



## https://www.hackerrank.com/challenges/sql-projects/problem
SELECT Start_Date, MIN(End_Date)
FROM 
    (SELECT Start_Date FROM Projects WHERE Start_Date NOT IN (SELECT End_Date FROM Projects)) a,
    (SELECT End_Date FROM Projects WHERE End_Date NOT IN (SELECT Start_Date FROM Projects)) b 
WHERE Start_Date < End_Date
GROUP BY Start_Date
ORDER BY DATEDIFF(MIN(End_Date), Start_Date) ASC, Start_Date ASC;

#https://www.hackerrank.com/challenges/placements/problem?h_r=next-challenge&h_v=zen
SELECT 
    s.name
FROM
    friends f
        LEFT JOIN
    packages p ON f.id = p.id
        LEFT JOIN
    packages p2 ON f.friend_id = p2.id
        LEFT JOIN
    students s ON f.id = s.id
WHERE
    p2.salary > p.salary
ORDER BY p2.salary


#https://www.hackerrank.com/challenges/symmetric-pairs/problem
SELECT f1.X, f1.Y FROM Functions f1
INNER JOIN Functions f2 ON f1.X=f2.Y AND f1.Y=f2.X
GROUP BY f1.X, f1.Y
HAVING COUNT(f1.X)>1 or f1.X<f1.Y
ORDER BY f1.X 