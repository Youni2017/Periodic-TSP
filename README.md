This competition is a programming contest organized by Prof. Sara Billey in the Spring 2024 quarter for the UW Math 381 class. The competition details are as follows:

# **Periodic TSP Competition**


# *Objective:* 

Given the coordinates for 128 cities in the United States (provided in 'cities2024.txt'), partition these cities into three distinct groups and determine an optimal route within each group. The goal is to minimize the total distance traveled for each group using a quadratic objective function as described in the Missouri Lottery paper (restricted to 3 days instead of 7).


# *Requirements:* 

1. City Partitioning and Route Planning:
   - Partition the 128 cities $C = G_1 \cup G_2 \cup G_3$ into three groups: $G_1, G_2,$ and $G_3$.
   - For each group, provide an ordered list of cities representing the route taken within that group.
   - Example: If the tour within $G1$ is $[0, 2, 7]$, then the total distance is calculated as: $d(v_0, v_2) + d(v_2, v_7) + d(v_7, v_0)$
   - You may use any method in Python or Gurobi to compute the optimal tour.

# *Submission Instructions:*
1. Python List of Routes:
   - In the body of your email, include a named Python list of 3 lists, representing the order in which you would visit the cities in each of the 3 groups.
   - Use your initials in the name of your list of lists.
   - Example:
     ``ScubaOrder = [range(0, 114), [115, 116], [117, 118, 119]]``
2. Attachment:
   - Attach the Python program and any output files used to generate your solution.
   - Name your file as TSPtourby.<your_name>.py or TSPtourby.<your_name>.ipynb.
3. Email Subject Line:
   - Use "Periodic TSP Competition" as the subject heading of your email.

