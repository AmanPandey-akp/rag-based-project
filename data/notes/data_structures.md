# Data Structures and Algorithms

## Time Complexity
Big-O notation describes how running time grows with input size n. Binary search is O(log n) and needs a sorted array. Merge sort is O(n log n) in every case and needs O(n) extra space. Quick sort is O(n log n) on average but O(n^2) in the worst case, for example on an already sorted array with a bad pivot choice; choosing a random pivot makes that unlikely. Heap sort is O(n log n) in place but not stable. Bubble sort and insertion sort are O(n^2), although insertion sort is fast on nearly sorted data.

## Hash Tables
A hash table maps keys to buckets using a hash function, giving average O(1) insert, delete and lookup. A collision happens when two keys map to the same bucket. Separate chaining stores colliding keys in a linked list per bucket. Open addressing keeps everything in the array and probes for the next free slot using linear probing, quadratic probing or double hashing. The load factor is the number of entries divided by the number of buckets; when it grows too high the table is resized and rehashed.

## Trees
A binary search tree keeps smaller keys on the left and larger keys on the right, so search is O(h) where h is the height. In the worst case, for example inserting sorted data, the tree degenerates into a linked list with h = n. Self-balancing trees fix this. An AVL tree keeps the height difference of the two subtrees of every node within 1 using rotations, so operations are O(log n). A red-black tree is less strictly balanced but needs fewer rotations on insertion. A heap is a complete binary tree where each parent is at least as large as its children (max-heap), used to implement priority queues. Tree traversals are inorder, preorder, postorder and level order; inorder traversal of a BST visits keys in sorted order.

## Graph Algorithms
Graphs are stored as an adjacency matrix, which uses O(V^2) space, or an adjacency list, which uses O(V + E) space and is better for sparse graphs. Breadth-first search uses a queue and finds the shortest path in an unweighted graph. Depth-first search uses a stack or recursion and is used for cycle detection and topological sorting. Dijkstra's algorithm finds the shortest paths from one source in a graph with non-negative edge weights, and runs in O((V + E) log V) with a binary heap. Bellman-Ford handles negative edge weights and detects negative cycles in O(VE). Kruskal's and Prim's algorithms build a minimum spanning tree.

## Dynamic Programming
Dynamic programming solves problems that have overlapping subproblems and optimal substructure by storing the results of subproblems so they are never recomputed. Top-down memoization caches recursive calls, and bottom-up tabulation fills a table iteratively. Classic examples are Fibonacci numbers, the 0/1 knapsack problem, the longest common subsequence and the edit distance between two strings.
