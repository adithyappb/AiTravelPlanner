import pandas as pd
from queue import Queue
import time
from numpy import False_

#----------------------------------------------------------------------------------#

#---------------------------------- Flight Class ----------------------------------#

#----------------------------------------------------------------------------------#
class Flight:
  def __init__(self, airline, origin_airport, destination_airport, scheduled_departure, scheduled_arrival, scheduled_time, distance):
    self.airline = airline
    self.origin = origin_airport
    self.destination = destination_airport
    self.departure_time = scheduled_departure
    self.arrival_time = scheduled_arrival
    self.time = scheduled_time
    self.distance = distance

  def __repr__(self) :
    st = self.origin + "->" + self.destination+", "+str(self.time) + " minutes, "+str(self.distance)+" miles"
    return st

#----------------------------------------------------------------------------------#

#---------------------------------- Airport Class ---------------------------------#

#----------------------------------------------------------------------------------#
class Airport:
  def __init__(self, code):
    self.code = code
    self.outboundFlights = []

  def addOutbound(self, flight) :
    self.outboundFlights.append(flight)
  
  def getShortestFlight(self) :
    if(len(self.outboundFlights)>0) :
      shortest = self.outboundFlights[0]
      for o in self.outboundFlights :
        if o.time < shortest.time :
          shortest = o
      return shortest

  def getCheapestFlight(self) :
    if(len(self.outboundFlights)>0) :
      cheapest = self.outboundFlights[0]
      for o in self.outboundFlights :
        if o.price < cheapest.price :
          cheapest = o
      return cheapest

  def __repr__(self) :
    return self.code

#----------------------------------------------------------------------------------#

#---------------------------------- Bellman Ford ----------------------------------#

#----------------------------------------------------------------------------------#
def BellmanFord(allFlights, airportCodes, startAirport, metric, dest):
    allDistances = [float("Inf")] * len(airportCodes)
    allDistances[airportCodes.index(startAirport)] = 0
    paths = [" : "] * len(airportCodes)
    for i in range(len(airportCodes)-1):
      for f in allFlights:
        originIndex = airportCodes.index(f.origin)
        destIndex = airportCodes.index(f.destination)
        m=0
        if metric=="1":
          m = f.distance
        else:
          m = f.time
        if allDistances[originIndex] != float("Inf") and allDistances[originIndex] + m < allDistances[destIndex]:
          allDistances[destIndex] = allDistances[originIndex] + m
          paths[destIndex] = paths[originIndex] + f.origin+ " -> "+ f.destination +" , "
    for f in allFlights:
      originIndex = airportCodes.index(f.origin)
      destIndex = airportCodes.index(f.destination)
      m=0
      if metric=="1":
        m = f.distance
      else:
        m = f.time
      if allDistances[originIndex] != float("Inf") and allDistances[originIndex] + m < allDistances[destIndex]:
        return 
   # printAll(metric, airportCodes, allDistances, paths)
    unit = ""
    if metric == "1":
      unit = "miles"
    elif metric == "2":
      unit = "hours"
    if dest == "":
     printBestAndWorst(airportCodes, allDistances, paths, unit)
    else:
      index = airportCodes.index(dest)
      print(airportCodes[index]+" : "+str(allDistances[index])+" "+unit+paths[index])

def printAll(metric, airportCodes, allDistances, paths):
  for i in range(0, len(allDistances)):
      if allDistances[i] != float("Inf"):
        unit = ""
        if metric == "1":
          unit = "miles"
        elif metric == "2":
          unit = "minutes"
        print(airportCodes[i]+" : "+str(allDistances[i])+" "+unit+paths[i])

def printBestAndWorst(airportCodes, allDistances, paths, unit):
    best = 0
    worst = 0
    for i in range(0, len(allDistances)):
      if allDistances[i] != float("Inf") and allDistances[i] != 0:
        if allDistances[i] < allDistances[best]:
          best = i
        if allDistances[i] > allDistances[worst]:
          worst = i
    bestTime = str(int(allDistances[best]/60))+":"+str(allDistances[best]%60)
    worstTime = str(int(allDistances[worst]/60))+":"+str(allDistances[worst]%60)
    print("Best Flight: "+airportCodes[best]+": "+bestTime+" "+unit+paths[best])
    print("Worst Flight: "+airportCodes[worst]+": "+worstTime+" "+unit+paths[worst])

#######################
#Start of Djikstra & A*
#######################
class Graph:
    def __init__(self, vert, edges):
        self.vert = vert
        self.edges = edges
        self.adj_list = {}
        self.edge_list = {}
        self.vert_list = {}
        for e in edges:
            self.adj_list[e] = []
            if e.u.ID not in self.adj_list:
                self.adj_list[e.u.ID] = []
            self.adj_list[e.u.ID].append(e.v.ID)
        for e in edges:
            self.edge_list[e.ID] = e
        for v in vert:
            self.vert_list[v.ID] = v


class Edge:
    def __init__(self, u, v, weight):
        self.u = u
        self.v = v
        self.weight = weight
        self.ID = self.u.ID + self.v.ID


class Vertex:
    def __init__(self, distance, predecessor, ID):
        self.distance = distance
        self.predecessor = predecessor
        self.ID = ID


# creates vertices from the list of vertex names
def init_vertices(vertex_list):
    vertex_final = []
    for i in range(len(vertex_list)):
        vertex_final.append(Vertex(float("inf"), None, vertex_list[i]))
    return vertex_final


# creates edges from the csv file
def create_edges_new(src, dest, cost, vertices, vert_list):
    edge_list = []
    for i in range(len(cost)):  # cost is good, need to find src and dest
        for j in range(len(vert_list)):
            if src[i] == vertices[j].ID:
                temp_src = vertices[j]
            if dest[i] == vertices[j].ID:
                temp_dest = vertices[j]
        edge_list.append(Edge(temp_src, temp_dest, cost[i]))
    return edge_list

def dijkstra(graph, start_node, target_node):
    unvisited_nodes = list(graph.vert)  # open list
    shortest_path = {}  # stores g value of nodes
    previous_nodes = {}  # closed list

    # this initializes the cost of the vertices to infinity and sets src cost to 0
    max_value = float("inf")
    for node in unvisited_nodes:
        shortest_path[node.ID] = max_value
    shortest_path[start_node] = 0
    # creates a counter for how many nodes were evaluted
    nodes_evaluated = 0
    while unvisited_nodes:  # while there are still nodes left to visit
        # increments the counter
        nodes_evaluated += 1
        # this selects the node from the array with the smallest distance
        min_node = None
        for node in unvisited_nodes:
            if min_node == None:
                min_node = node
            elif shortest_path[node.ID] < shortest_path[min_node.ID]:
                min_node = node
        # terminate if the shortest path candidate is the target destination
        if min_node == get_src(graph, target_node):
            print("Dijkstra nodes evaluated = " + str(nodes_evaluated))
            return previous_nodes, shortest_path
        if (
            min_node.ID in graph.adj_list
        ):  # checking to make sure the node has out edges 
            # iterating through neighbors to update distances
            neighbors = graph.adj_list[min_node.ID]
            for neighbor in neighbors:
                tentative_value = (
                    shortest_path[min_node.ID]
                    + graph.edge_list[min_node.ID + neighbor].weight
                )
                # storing the node path and distances in the appropriate dictionaries
                if tentative_value < shortest_path[neighbor]:
                    shortest_path[neighbor] = tentative_value
                    previous_nodes[neighbor] = min_node.ID
        # remove the evaluated node from the pool
        unvisited_nodes.remove(min_node)
    # Displayed the counter of how many nodes were evaluated
    print("-------------------------------------")
    print("Dijkstra nodes evaluated = " + str(nodes_evaluated))
    # return the shortest path and node path
    return previous_nodes, shortest_path


# returns a vertex given the vertex ID
def get_src(graph, node_id):
    node = graph.vert_list[node_id]
    return node


def bfs(graph, start_node, target_node):
    # initializes node list and assigns appropriate values
    total_nodes = list(graph.vert)
    max_value = float("inf")
    for node in total_nodes:
        node.distance = max_value
    s = get_src(graph, start_node)
    s.distance = 0
    # creates queue and puts src in queue
    pending_nodes = Queue()
    pending_nodes.put(s)

    while not pending_nodes.empty():
        # gets next node from queue
        node = pending_nodes.get()
        if node == get_src(graph, target_node):
            return graph
        if node.ID in graph.adj_list:  # makes sure the node has out edges
            # assigns neighbors as the out edges of the current node
            neighbors = graph.adj_list[node.ID]
            # iterates through neighbors
            for neighbor in neighbors:
                next_node = graph.edge_list[node.ID + neighbor].v
                # updates neighbors distance value and predecessors
                if next_node.distance == max_value:
                    next_node.distance = node.distance + 1
                    next_node.predecessor = node
                    # calls the next node into the queue to be evaluated
                    pending_nodes.put(next_node)
    # returns the graph that the values can be derived from
    return graph


# returns the number of hops from a start node to an end node(used as h(x))
def h_x_bfs_distance(graph, start_node, target_node):
    graph = bfs(graph, start_node, target_node)
    hops = get_src(graph, target_node).distance
    return hops


def a_starSearch(graph, bfs_graph, start_node, target_node, min_cost):
    # initializes list of nodes and dictionaries that will record the path and score
    unvisited_nodes = list(graph.vert)
    shortest_path = {}  # gscore
    fscore = {}  # fscore
    previous_nodes = {}
    # initializes the values of the nodes
    max_value = float("inf")
    for node in unvisited_nodes:
        shortest_path[node.ID] = max_value
        fscore[node.ID] = max_value
        node.distance = max_value
    shortest_path[start_node] = 0
    fscore[start_node] = 0
    get_src(graph, start_node).distance = 0
    # a separate graph is used for bfs, because running bfs will change the values of the predecessors and the path
    get_src(bfs_graph, start_node).distance = 0
    # creates a counter for how many nodes were evaluated
    nodes_evaluated = 0
    while unvisited_nodes:
        # increments the counter
        nodes_evaluated += 1
        # selecting the node with the smallest fscore
        min_node = None
        for node in unvisited_nodes:
            if min_node == None:
                min_node = node
            elif fscore[node.ID] < fscore[min_node.ID]:
                min_node = node
        # terminate if the shortest path candidate is the target destination
        if min_node == get_src(graph, target_node):
            print("-------------------------------------")
            print("A* nodes evaluated = " + str(nodes_evaluated))
            return previous_nodes, shortest_path

        elif (
            min_node.ID in graph.adj_list
        ):  # making sure that there are outgoing edges from this node
            neighbors = graph.adj_list[
                min_node.ID
            ]  # list of 'v' nodes on outgoing edges from min_node

            for neighbor in neighbors:  # iterating through all of the neighbors
                tentative_value = (
                    shortest_path[min_node.ID]
                    + graph.edge_list[min_node.ID + neighbor].weight
                )  # this is g_x
                h_x = (
                    h_x_bfs_distance(bfs_graph, neighbor, target_node) * min_cost
                )  # calculating minimum hops from neighbor to destination or the heuristic value h(x)
                f_x = tentative_value + h_x  # adding minimum hops to current distance

                if (
                    tentative_value < shortest_path[neighbor]
                ):  # if the new value is shorter than the previously calculated one
                    shortest_path[
                        neighbor
                    ] = tentative_value  # update the distance of the shortest path
                    fscore[neighbor] = f_x
                    previous_nodes[neighbor] = min_node.ID  # add this to the path

        unvisited_nodes.remove(min_node)  # remove it from the list of nodes to visit
    # prints the counter
    print("A* nodes evaluated = " + str(nodes_evaluated))
    return previous_nodes, shortest_path

# print_result is used to display the distance and node path for dijkstra's and A* algorithm
def print_result(previous_nodes, shortest_path, src, dest):
    path = []
    node = dest
    while node != src:
        path.append(node)
        node = previous_nodes[node]
    path.append(src)
    print("-------------------------------------")
    print(
        "The least distance path covered in miles(mi) is : "
        + str(round(shortest_path[dest], 2))
    )
    print(" -> ".join(reversed(path)))

'''
  STARTING POINT
'''
#Reading the data from the csv file
flight_data = pd.read_csv('flightData.csv')
nodes = pd.read_csv(r'Nodes.csv')

for i in flight_data.columns:
    flight_data.fillna(0, inplace=True)

arrayDataFlights = flight_data.to_numpy()

allAirports = []
allFlights = []
airportCodes = []

# Creates flight array, airport codes array, and airports array
for a in arrayDataFlights :
  allFlights.append(Flight(a[4], a[7], a[8], a[9], a[20], a[14], a[17]))
  if a[7] not in airportCodes:
    airportCodes.append(a[7])
    allAirports.append(Airport(a[7]))
  if a[8] not in airportCodes:
    airportCodes.append(a[8])
    allAirports.append(Airport(a[8]))

# Adds outbound flights to airport objects
for f in allFlights :
  for a in allAirports :
    if f.origin == a.code :
      a.addOutbound(f)

#sorting the data into separate arrays
src_list=flight_data['ORIGIN_AIRPORT'].values
dest_list = flight_data['DESTINATION_AIRPORT'].values
cost_list = flight_data['DISTANCE'].values
vert_list = nodes['Nodes'].values

#finding the cheapest flight in the dataset to use in the heuristic algorithm
min_cost = float('inf')
for i in range(len(cost_list)):
    if cost_list[i] < min_cost:
        min_cost = cost_list[i]


#converting the name strings from the data into Vertex types
vertices = init_vertices(vert_list)
#creating Edge types from teh data
edges = create_edges_new(src_list, dest_list, cost_list, vertices, vert_list)

#creating the graphs that will be input into A*
standard_graph = Graph(vertices, edges)
bfs_graph = Graph(vertices, edges)

#----------------------------------------------------------------------------------#

#------------------------------------- User Input ---------------------------------#

#----------------------------------------------------------------------------------#
# Print airport codes
for i in range(0, len(airportCodes)-10, 10):
  print(airportCodes[i : i+10])

# Input loop
while True:
  print("")
  print("Press E to Exit")

# Enter origin airport
  airport = input("Origin airport code: ")
  while airport not in airportCodes and airport!="E":
    print('Invalid input')
    airport = input("please choose a valid airport code: ")
  if airport == "E":
    break

# Enter destination airport
  dest = input("Enter destination airport, or 'A' for anywhere: ")
  while dest not in airportCodes and dest!="A" and dest!="E":
    print('Invalid input')
    dest = input("Enter destination airport, or 'A' for anywhere: ")
  if dest == "E":
    break

# Enter metric type
  metric = input("Distance (1) or Time (2)? : ")
  while metric != "1" and metric != "2" and metric !="E":
    metric = input("Please choose a valid metric: ")
  if metric == "E":
    break

# Algorithm calls
  print("")
  if dest=="A":
    start_bm = time.time()
    print("The distance covered to - ")
    BellmanFord(allFlights, airportCodes, airport, metric, "")
    end_bm = time.time()
    #displaying results
    print("Time Taken for BellmanFord : ", end_bm - start_bm)
    print("Start : ", start_bm)
    print("Stop : ", end_bm)
    print("-------------------------------------")

  else:
    # Other algorithm calls : Djikstra & A*
    start_bm = time.time()
    print("The distance covered to - ")
    BellmanFord(allFlights, airportCodes, airport, metric, dest)
    end_bm = time.time()
    #displaying results
    print("\nTime Taken for BellmanFord : ", end_bm - start_bm)
    print("Start : ", start_bm)
    print("Stop : ", end_bm)
    print("-------------------------------------")

    #Assigning the src and destination nodes
    #start = "HNL"
    #end = "BDL"
    start = airport
    end = dest

    start_aS = time.time()
    #Call the A* algorithm
    prev_nodes, path = a_starSearch(standard_graph, bfs_graph, start, end, min_cost)

    end_aS = time.time()

    #displaying results
    print_result(prev_nodes, path, start, end)
    print("Time Taken for A* : ", end_aS - start_aS)
    print("Start : ", start_aS)
    print("Stop : ", end_aS)
    print("-------------------------------------")

    #Assigning the src and destination nodes
    #start = "HNL"
    #end = "BDL"
    #[HNL,BDL], [ANC,JAX]
    start = airport
    end = dest

    start_dj = time.time()
    #Call Dijkstra's algorithm
    previous_nodes, shortest_path = dijkstra(standard_graph, start, end)
    end_dj = time.time()


    # displaying results
    print_result(previous_nodes, shortest_path, start, end)
    print(end_dj - start_dj)
    print("Time Taken for Dijkstra : ", end_dj - start_dj)
    print("Start : ", start_dj)
    print("Stop : ", end_dj)
