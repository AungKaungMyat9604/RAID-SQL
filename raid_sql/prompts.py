"""RAID-SQL v2 slim format anchors (RAG supplies real few-shots).

Legacy DIN banks live in ``prompts_din.py`` behind ``USE_DIN_STATIC_BANKS``.
"""

schema_linking_prompt = '''Q: "Find the buildings which have rooms with capacity more than 50."
A: Let’s think step by step. We need classroom.building and classroom.capacity; cell value 50.
Schema_links: [classroom.building,classroom.capacity,50]

Q: "How many heads of the departments are older than 56 ?"
A: Let’s think step by step. We need head.* and head.age; cell value 56.
Schema_links: [head.*,head.age,56]

Q: "List the id of students who never attends courses?"
A: Let’s think step by step. We need Students.student_id and Student_Course_Attendance.student_id with FK.
Schema_links: [Students.student_id = Student_Course_Attendance.student_id]

'''

classification_prompt = '''Q: "Find the buildings which have rooms with capacity more than 50."
schema_links: [classroom.building,classroom.capacity,50]
A: Let’s think step by step. Needs tables = [classroom], no JOIN, no nested (INTERSECT/UNION/EXCEPT/IN/NOT IN). Sub-questions = [""].
Label: "EASY"

Q: "What are the names of all instructors who advise students in the math depart sorted by total credits of the student."
schema_links: [advisor.i_id = instructor.id,advisor.s_id = student.id,instructor.name,student.dept_name,student.tot_cred,math]
A: Let’s think step by step. Needs tables = [advisor,instructor,student], JOIN, no nested. Sub-questions = [""].
Label: "NON-NESTED"

Q: "How many courses that do not have prerequisite?"
schema_links: [course.*,course.course_id = prereq.course_id]
A: Let’s think step by step. Needs nested (NOT IN / EXCEPT). Sub-questions = ["Which courses have prerequisite?"].
Label: "NESTED"

Q: "Find the title of course that is provided by both Statistics and Psychology departments."
schema_links: [course.title,course.dept_name,Statistics,Psychology]
A: Let’s think step by step. Needs INTERSECT for "both". Sub-questions = ["Find titles provided by Psychology"].
Label: "NESTED"

Q: "Find the id of instructors who taught a class in Fall 2009 but not in Spring 2010."
schema_links: [teaches.id,teaches.semester,teaches.year,Fall,2009,Spring,2010]
A: Let’s think step by step. Needs EXCEPT. Sub-questions = ["Instructors who taught in Spring 2010"].
Label: "NESTED"

Q: "Give the name and building of the departments with greater than average budget."
schema_links: [department.budget,department.dept_name,department.building]
A: Let’s think step by step. Needs nested avg subquery. Sub-questions = ["What is the average budget"].
Label: "NESTED"

'''

easy_prompt = '''Q: "Find the buildings which have rooms with capacity more than 50."
Schema_links: [classroom.building,classroom.capacity,50]
SQL: SELECT DISTINCT building FROM classroom WHERE capacity > 50

Q: "How many singers are from each country?"
Schema_links: [singer.Country,singer.*]
SQL: SELECT Country , COUNT(*) FROM singer GROUP BY Country

Q: "Find the room number of the rooms which can sit 50 to 100 students and their buildings."
Schema_links: [classroom.building,classroom.room_number,classroom.capacity,50,100]
SQL: SELECT building , room_number FROM classroom WHERE capacity BETWEEN 50 AND 100

'''

medium_prompt = '''Q: "Find the name and building of the department with the highest budget."
Schema_links: [department.budget,department.dept_name,department.building]
A: Let’s think step by step. Join tables = []. Intermediate_representation: select dept_name , building order by budget desc limit 1
SQL: SELECT dept_name , building FROM department ORDER BY budget DESC LIMIT 1

Q: "Find the title of courses that have two prerequisites?"
Schema_links: [course.title,course.course_id = prereq.course_id]
A: Let’s think step by step. Join tables = [course,prereq]. Use Foreign_keys for JOIN.
SQL: SELECT T1.title FROM course AS T1 JOIN prereq AS T2 ON T1.course_id = T2.course_id GROUP BY T2.course_id HAVING count(*) = 2

Q: "list in alphabetic order all course names and their instructors' names in year 2008."
Schema_links: [course.title,course.course_id = teaches.course_id,teaches.id = instructor.id,instructor.name,teaches.year,2008]
A: Let’s think step by step. Join tables = [course,teaches,instructor]. Match FK columns carefully.
SQL: SELECT T1.title , T3.name FROM course AS T1 JOIN teaches AS T2 ON T1.course_id = T2.course_id JOIN instructor AS T3 ON T2.id = T3.id WHERE T2.YEAR = 2008 ORDER BY T1.title

'''

hard_prompt = '''Q: "Find the name of the courses that do not have any prerequisite?"
Schema_links: [course.title,course.course_id]
A: Let's think step by step. Sub-question "courses that have any prerequisite?".
The SQL query for the sub-question is SELECT course_id FROM prereq
So, the answer to the question is =
SQL: SELECT title FROM course WHERE course_id NOT IN (SELECT course_id FROM prereq)

Q: "Find the id of instructors who taught a class in Fall 2009 but not in Spring 2010."
Schema_links: [teaches.id,teaches.semester,teaches.YEAR,Fall,2009,Spring,2010]
A: Let's think step by step. Sub-question "instructors who taught in Spring 2010".
The SQL query for the sub-question is SELECT id FROM teaches WHERE semester = 'Spring' AND YEAR = 2010
So, the answer to the question is =
SQL: SELECT id FROM teaches WHERE semester = 'Fall' AND YEAR = 2009 EXCEPT SELECT id FROM teaches WHERE semester = 'Spring' AND YEAR = 2010

Q: "Find the title of course that is provided by both Statistics and Psychology departments."
Schema_links: [course.title,course.dept_name,Statistics,Psychology]
A: Let's think step by step. Sub-question "titles provided by Psychology".
The SQL query for the sub-question is SELECT title FROM course WHERE dept_name = 'Psychology'
So, the answer to the question is =
SQL: SELECT title FROM course WHERE dept_name = 'Statistics' INTERSECT SELECT title FROM course WHERE dept_name = 'Psychology'

Q: "Show the status shared by cities with population bigger than 1500 and smaller than 500."
Schema_links: [city.Status,city.Population,1500,500]
A: Let's think step by step. Sub-question "statuses of cities with population < 500".
The SQL query for the sub-question is SELECT Status FROM city WHERE Population < 500
So, the answer to the question is =
SQL: SELECT Status FROM city WHERE Population > 1500 INTERSECT SELECT Status FROM city WHERE Population < 500

'''

DEBUG_INSTRUCTION = """#### For the given question, use the provided tables, columns, foreign keys, and primary keys to fix the given SQLite SQL QUERY for any issues. If there are any problems, fix them. If there are no issues, return the SQLite SQL QUERY as is.
#### Use the following instructions for fixing the SQL QUERY:
1) Use the database values that are explicitly mentioned in the question.
2) Pay attention to the columns that are used for the JOIN by using the Foreign_keys.
3) Use DESC when needed. If the question says "distinct", use SELECT DISTINCT.
4) Pay attention to the columns that are used for the GROUP BY statement.
5) Pay attention to the columns that are used for the SELECT statement.
6) Only change the GROUP BY clause when necessary (Avoid redundant columns in GROUP BY).
7) Use GROUP BY on one column only.
8) If the question asks for values shared by / belonging to BOTH of two conditions (e.g. "both A and B", dual thresholds like "more than X and less than Y"), use INTERSECT (or equivalent set overlap). Do NOT use OR, and do NOT keep only one side of the condition.
9) Emit a complete executable SQL statement with balanced parentheses. Never return only a subquery fragment.
10) SELECT column order matters for official EX. Put aggregates (COUNT/SUM/AVG/MIN/MAX) FIRST when the question leads with how many / number of / count the number / what is the number / find the average WITHOUT "each", then the GROUP BY key. If the question says "each" / "for each", put the GROUP BY key BEFORE the aggregate. Otherwise follow attribute order in the question.
11) For multi-table questions, JOIN using Foreign_keys; do not invent join keys.
12) Match inequality direction to the question: "not more than / at most N" → <= N; "more than N" → > N (HAVING/WHERE).

"""

EG_REPAIR_INSTRUCTION = """#### The SQLite engine failed, returned an unexpected empty result, or the SQL is incomplete / semantically mismatched for the question.
#### Fix the SQL using the schema, foreign keys, primary keys, sample DB values, and the SQLite error message.
#### Return ONLY the fixed SQL starting after the SELECT keyword line marker.
Rules:
1) Keep intent of the question.
2) Fix syntax, missing columns, bad JOINs, and GROUP BY issues.
3) Prefer values that appear in sample DB values / the question.
4) Output a single executable SQLite query body (continuation after SELECT).
5) Complete truncated or unbalanced SQL (missing SELECT arms, stray ')').
6) For "both / shared by / A and B" overlap questions, use INTERSECT (not OR, not a single predicate).
7) SELECT order: metric-first WITHOUT "each" → AGG then key; How many … each → key then AGG; else follow the question's attribute order.
8) When repairing JOINs, use exactly the Foreign_keys listed; fix WHERE / GROUP BY to match the question.
9) If the question says "distinct", emit SELECT DISTINCT.
10) Honor "not more than / at most / more than" thresholds in HAVING/WHERE.
\n"""
