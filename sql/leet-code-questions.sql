LEET CODE QUESTIONS

#Get Nth highest salary
CREATE FUNCTION getNthHighestSalary(N INT) RETURNS INT
BEGIN
declare M INT;
SET M = N-1;
  RETURN (
      # Write your MySQL query statement below.
      
      select Salary from Employee ORDER BY salary ASC limit 1 offset M
  );
END

