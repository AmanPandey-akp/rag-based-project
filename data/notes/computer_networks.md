# Computer Networks

## OSI and TCP/IP Models
The OSI model has seven layers: Physical, Data Link, Network, Transport, Session, Presentation and Application. The practical TCP/IP model merges these into four: Link, Internet, Transport and Application. Each layer adds its own header to the data, which is called encapsulation. IP at the network layer handles logical addressing and routing, while Ethernet at the data link layer moves frames between devices on the same network using MAC addresses.

## TCP vs UDP
TCP is connection-oriented and reliable. It uses a three-way handshake (SYN, SYN-ACK, ACK) to establish a connection, sequence numbers and acknowledgements for reliable in-order delivery, flow control with a sliding window, and congestion control. UDP is connectionless and has no delivery guarantee, ordering or congestion control, so it has very low overhead. TCP is used for web pages, email and file transfer; UDP is used for DNS queries, video streaming and online games where speed matters more than perfect delivery.

## TCP Congestion Control
TCP controls how fast it sends using a congestion window (cwnd). In slow start the window doubles every round trip until it reaches the slow-start threshold. After that, congestion avoidance grows it by one segment per round trip. When packet loss is detected through a timeout, the threshold is cut to half the window and the window drops back to one segment. Fast retransmit resends a segment after three duplicate acknowledgements, without waiting for the timeout, and fast recovery avoids dropping all the way back to slow start.

## IP Addressing and Subnetting
An IPv4 address is 32 bits long, written as four decimal octets. CIDR notation such as 192.168.10.0/24 says that the first 24 bits identify the network. A /24 network has 256 addresses, of which 254 are usable for hosts because the first is the network address and the last is the broadcast address. Subnetting borrows host bits to create smaller networks; a /26 gives four subnets of 64 addresses each. IPv6 uses 128-bit addresses to solve the exhaustion of IPv4 space.

## DNS and HTTP
The Domain Name System translates names like example.com into IP addresses. A resolver asks a root server, then a top-level domain server, then the authoritative server for the domain, and caches the answer for the time given by its TTL. HTTP is the application-layer request-response protocol of the web. HTTP is stateless, so cookies and sessions are used to remember users. HTTPS is HTTP over TLS, which provides encryption, integrity and server authentication using certificates.

## Routing
Routers forward packets using a routing table. Distance-vector protocols such as RIP use the Bellman-Ford idea, where each router tells its neighbours what it knows, and they can suffer from the count-to-infinity problem. Link-state protocols such as OSPF flood the whole topology to every router, and each router runs Dijkstra's algorithm to compute shortest paths. BGP is the path-vector protocol that connects different autonomous systems on the internet.
