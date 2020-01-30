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