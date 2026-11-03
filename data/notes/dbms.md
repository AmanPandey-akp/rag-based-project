# Database Management Systems

## ACID Properties
A transaction is a logical unit of work that must leave the database consistent. ACID stands for four guarantees. Atomicity means all operations of a transaction happen or none do; the recovery manager uses the undo log to roll back partial work. Consistency means a transaction moves the database from one valid state to another, preserving integrity constraints. Isolation means concurrent transactions behave as if they ran one after another. Durability means once a transaction commits, its changes survive crashes, which is achieved with write-ahead logging.

## Normalization
Normalization removes redundancy and update anomalies by decomposing tables. First Normal Form (1NF) requires atomic column values with no repeating groups. Second Normal Form (2NF) is 1NF plus no partial dependency, meaning no non-key attribute depends on only part of a composite key. Third Normal Form (3NF) is 2NF plus no transitive dependency, so non-key attributes depend only on the key. Boyce-Codd Normal Form (BCNF) is stricter: for every functional dependency X -> Y, X must be a superkey. Over-normalizing can force many joins, so read-heavy systems sometimes denormalize on purpose.

## Indexing and B+ Trees
An index speeds up lookups at the cost of extra storage and slower writes. Most relational databases use B+ trees. In a B+ tree all records live in the leaf nodes, and the leaves are linked in a list, which makes range queries such as BETWEEN very fast. The tree stays balanced, so search takes O(log n) disk accesses. A clustered index decides the physical order of rows, so a table can have only one. A non-clustered index is a separate structure holding pointers to the rows. Hash indexes give O(1) equality lookups but cannot answer range queries.

## Concurrency Control and Isolation Levels
Without control, concurrent transactions cause dirty reads, non-repeatable reads and phantom reads. SQL defines four isolation levels. Read Uncommitted allows dirty reads. Read Committed prevents dirty reads. Repeatable Read also prevents non-repeatable reads. Serializable prevents all anomalies including phantoms but reduces throughput. Two-phase locking (2PL) is a classic protocol: a growing phase acquires locks, a shrinking phase releases them, and no lock is acquired after the first release. Strict 2PL holds exclusive locks until commit, which avoids cascading aborts. Multiversion concurrency control (MVCC) lets readers see a snapshot so they never block writers.

## SQL Joins
An INNER JOIN returns only rows that match in both tables. A LEFT JOIN returns all rows from the left table and matching rows from the right, filling unmatched right columns with NULL. A RIGHT JOIN is the mirror image. A FULL OUTER JOIN returns all rows from both sides. A CROSS JOIN returns the Cartesian product of the two tables.
