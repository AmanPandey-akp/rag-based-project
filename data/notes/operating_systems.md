# Operating Systems

## Processes and Threads
A process is a program in execution with its own address space, registers and open files. A thread is a lightweight unit of execution inside a process. Threads of one process share the code, heap and open files but each has its own stack and program counter, so creating a thread and switching between threads is cheaper than for processes. A context switch saves the state of the running process in its Process Control Block (PCB) and loads the state of the next one.

## CPU Scheduling
The scheduler decides which ready process runs next. First Come First Served (FCFS) is simple but suffers from the convoy effect, where short jobs wait behind a long one. Shortest Job First (SJF) minimizes average waiting time but needs to know burst times and can starve long jobs. Round Robin gives each process a fixed time quantum and is fair for time-sharing systems; a very large quantum degenerates into FCFS while a very small one wastes time on context switches. Priority scheduling can starve low-priority processes, and aging, which gradually raises the priority of waiting processes, fixes that.

## Deadlocks
A deadlock is a situation where a set of processes wait forever for resources held by each other. Four Coffman conditions must hold together: mutual exclusion, hold and wait, no preemption, and circular wait. Breaking any one of them prevents deadlock. For example, forcing processes to request all resources at once breaks hold and wait, and imposing a global ordering on resources breaks circular wait. Deadlock avoidance uses the Banker's algorithm, which grants a request only if the system stays in a safe state. Detection builds a wait-for graph and looks for cycles, and recovery kills a process or preempts resources.

## Memory Management and Paging
Paging divides logical memory into fixed-size pages and physical memory into frames of the same size, which removes external fragmentation. The page table maps page numbers to frame numbers. Because a page table lookup costs an extra memory access, a Translation Lookaside Buffer (TLB), a small fast cache of recent translations, is used. A page fault occurs when the page is not in memory and must be loaded from disk. Page replacement algorithms choose the victim: FIFO is simple but suffers from Belady's anomaly, LRU evicts the least recently used page and performs well, and the optimal algorithm evicts the page used farthest in the future but cannot be implemented. Thrashing happens when a system spends more time paging than executing.

## Synchronization
A race condition occurs when the result depends on the order in which threads access shared data. The critical section problem requires mutual exclusion, progress and bounded waiting. A semaphore is an integer variable accessed only through wait (P) and signal (V) operations. A binary semaphore acts like a lock, while a counting semaphore manages a pool of identical resources. A mutex is a lock that must be released by the thread that acquired it. The classic problems are producer-consumer, readers-writers and dining philosophers.
