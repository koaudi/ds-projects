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