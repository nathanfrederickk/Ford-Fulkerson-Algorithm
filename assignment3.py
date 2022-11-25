
#-----------------------------------------------------------------------------------------------------------------------
from collections import deque
import math

class Vertex:
    def __init__(self, id, demand) -> None:
        """
        Initialize a vertex and its attributes. These vertex class will then be used to
        represent housemates, a restaurant, meals, as well as source and destination.

        :Input:
        id      : an integer which is then the vertex id
        demand  : an integer which is the demand of the vertex

        :Output, return or postcondition: Initialize a vertex

        :Time complexity: O(1)
        :Aux space complexity: O(1), since constant extra space needed.
        """
        #O(1) time and aux since it's all initialization
        self.id = id
        self.edges = []
        self.previous = None
        self.visited = False
        self.discovered = False
        self.demand = demand

    def add_edge(self, u, v, flow, maximum, lower, type = False, available = True):
        """
        Assign a directed edge with attributes of source, destination, flow, maximum flow, type,  
        and the availability of the edge to a vertex by appending the edges to the instance
        variable self.edges in each vertex.

        :Input:
        u           : a vertex which is the origin of the edge
        v           : a vertex which is the end of the edge.
        flow        : an integer to determine the current flow from u to v.
        maximum     : an integer to determine the maximum/ upper bound flow from u to v.
        lower       : an integer to determine the minimal/ lower bound flow from u to v.
        type        : a boolean to determine the type of the edges. True means it is reversed,
                      while False indicates that it is not reversed. (Used mainly in residual network)
        available   : a boolean to determine if the edge is able to be used in a bfs. True means
                      it is able, while False means otherwise.

        :Output, return or postcondition: Edge added onto the vertex

        :Time complexity: O(1)
        :Aux space complexity: O(1), since constant extra space is needed
        """
        self.edges.append(Edge(u, v, flow, maximum, lower, type, available))

    def __str__(self):
        return_string = str(self.id)
        return return_string

class Edge:
    def __init__(self, u: Vertex, v: Vertex, flow, maximum, lower, type = False, available = True):
        """
        Initialize a directed edge with u being the starting point, v being the end point,
        flow be the current flow from u to v, maximum is the upper bound from u to v,
        lower is the lower bound from u to v, type determines if the edge is reversed or not,
        and available determines if it can be used in the bfs or not.

        :Input:
        u           : a vertex which is the origin of the edge
        v           : a vertex which is the end of the edge.
        flow        : an integer to determine the current flow from u to v.
        maximum     : an integer to determine the maximum/ upper bound flow from u to v.
        lower       : an integer to determine the minimal/ lower bound flow from u to v.
        type        : a boolean to determine the type of the edges. True means it is reversed,
                      while False indicates that it is not reversed. (Used mainly in residual network)
        available   : a boolean to determine if the edge is able to be used in a bfs. True means
                      it is able, while False means otherwise.

        :Output, return or postcondition: Initializes a directed edge from u to v
                                          with attributes required for a flow network.

        :Time complexity: O(1) since it's constant
        :Aux space complexity: O(1), since constant extra space is needed
        """
        self.u = u
        self.v = v
        self.flow = flow
        self.lower = lower
        self.maximum = maximum
        self.type = type
        self.available = available

    def __str__(self):
        return_string = "(" + str(self.u) + ", "+ str(self.v) + ")"
        return return_string

class Graph:
    def __init__(self, availability) -> None:
        """
        Initialize a Graph from the input availability. The graph designs a flow network
        with a start, 5 housemates, 1 restaurant, the number of meals which is 2*len(availability), and
        the end vertex. The class connects an edge from housemates to the meals according to their mode.
        If the mode is 0, then no edge is formed between the breakfast and dinner of that day, if it is 1, 
        the particular housemate will only be connected to the breakfast of that day, if it is 2 will only be connected
        to the dinner of that day, and lastly, if it is 3, it will create a medium node between the housemate and
        the meals.

        :Input:
        availability    : a nested list of integers where len(availability) determines the number of days,
                          and each integer in the nested list determines the index's preference of preparing a 
                          meal on that day. If it is 0, the person would like not to make a meal, 1 is only breakfast,
                          2 is only dinner, and 3 if the person doesn't mind doing either one.

        :Output, return or postcondition: Initializes a flow network design with lower bounds, upper
                                          bounds, start, housemates, restaurant, meals, and end vertex. This class
                                          also adds the middle nodes for specific preference of the housemates.

        :Time complexity: O(n) where n is the number of days, or len(availability)
        :Aux space complexity: O(n)
        """
        days = len(availability)

        self.meals = days * 2
        self.lower_bound = math.floor(0.36*days)
        self.upper_bound = math.ceil(0.44*days)

        # The number of housemates + restaurant is always 6
        people_cafe = 6

        # The number of meals + housemates + restaurant
        total = self.meals + people_cafe

        # Initialize the start, housemates, restaurant, and meals vertices.
        # The start vertex is 0, housemates and restaurants are from 1-6, and the meals are from index 7::
        # O(2n + 6 + 2) which is dominated by O(n)
        self.vertices = [Vertex(0, 0)] + list(Vertex(location, 0) for location in range(1, total + 1))

        # Connect the edges from the start to the housemates.
        # O(1)
        for person in range(1, 6):
            self.vertices[0].add_edge(self.vertices[0], self.vertices[person], 0, self.upper_bound, self.lower_bound)

        # Connect the edges from the start to the restaurant.
        # O(1)
        self.vertices[0].add_edge(self.vertices[0], self.vertices[6], 0, math.floor(0.1*days), 0)


        counter = 7

        # To connect the preference of each housemates to each meal(s) in each day
        # O(n)
        for day in range(len(availability)):

            # O(n), since it enters the loop O(5n) times
            for mode in range(len(availability[day])):
                # If mode is 0, it doesn't connect the housemate to a meal in the day
                if availability[day][mode] == 0:
                    continue

                # If mode is 1, it connects the housemate into the breakfast of the current day
                elif availability[day][mode] == 1:
                    self.vertices[mode + 1].add_edge(self.vertices[mode + 1], self.vertices[counter + day], 0, 1, 0)

                # If mode is 2, it connects the housemate into the dinner of the current day
                elif availability[day][mode] == 2:
                    self.vertices[mode + 1].add_edge(self.vertices[mode + 1], self.vertices[counter + day + 1], 0, 1, 0)
                
                # If mode is 3, it connects the housemate into a middle node with lower bound of 0 and upper bound
                # of 1, then connects the middle node to the breakfast and dinner with each edge lower bound of 0 
                # and upper bound of 1
                else:
                    self.vertices.append(Vertex(len(self.vertices), 0))
                    self.vertices[mode + 1].add_edge(self.vertices[mode + 1], self.vertices[-1], 0, 1, 0)
                    self.vertices[-1].add_edge(self.vertices[-1], self.vertices[counter + day], 0, 1, 0)
                    self.vertices[-1].add_edge(self.vertices[-1], self.vertices[counter + day + 1], 0, 1, 0)

            # Connects the restaurant to each meal of the day since the restaurant food can be ordered within any day
            self.vertices[6].add_edge(self.vertices[6], self.vertices[counter + day], 0, 1, 0)
            self.vertices[6].add_edge(self.vertices[6], self.vertices[counter + day + 1], 0, 1, 0)

            counter += 1
        
        # Creates the end vertex 
        # O(1)
        self.vertices.append(Vertex(len(self.vertices), 0))
        
        # Connects the meals to the end vertex
        # O(2n) which is O(n)
        for meal in range(7, 7 + self.meals):
            self.vertices[meal].add_edge(self.vertices[meal], self.vertices[-1], 0, 1, 1) 

        # Connects the end vertex to the start vertex with the max flow of the number of meals
        # O(1)
        self.vertices[-1].add_edge(self.vertices[-1], self.vertices[0], 0, self.meals, 0)

    def min_flow(self):
        """
        Eliminates the lower bound from each edge and sets the demand of each edge.

        :Output, return or postcondition: A graph with a lower bound of zero and a vertex
                                          with each demand.

        :Time complexity: O(n) where n is the number of days, or len(availability)
        :Aux space complexity: O(n)
        """
        # sets the demand of the start vertex and the housemates
        # O(1)
        for edge in self.vertices[0].edges:
            u = edge.u
            v = edge.v
            lower = edge.lower

            u.demand += lower
            v.demand -= lower
            edge.maximum -= lower 

        # sets the demand of the meals and the end vertex
        # O(n)
        for meal in range(7, 7 + self.meals):
            for edge in self.vertices[meal].edges:
                u = edge.u
                v = edge.v
                lower = edge.lower

                u.demand += lower
                v.demand -= lower
                edge.maximum -= lower

    def optimize(self):
        """
        Creates a source vertex and connect edges to vertices with a negative demand with a
        maximum flow of its abs(demand), and creates a sink vertex which all the vertices with
        positive demand will connect into. The max flow will be its demand.
        
        :Output, return or postcondition: A graph with source connecting into all the vertex with negative demand
                                          and sink that is connected into a vertex with positive demand. The max
                                          flow of each edge is the abs(demand) of the vertex.

        :Time complexity: O(n) where n is the number of days, or len(availability)
        :Aux space complexity: O(n)
        """
        # Creates a source
        self.vertices.append(Vertex(len(self.vertices), 0))
        # O(1)
        for i in range(1,6):
            self.vertices[-1].add_edge(self.vertices[-1], self.vertices[i], 0, abs(self.vertices[i].demand), 0) 

        # connects the source to the end vertex
        self.vertices[-1].add_edge(self.vertices[-1], self.vertices[-2], 0, abs(self.vertices[-2].demand), 0) 

        # Creates a sink
        self.vertices.append(Vertex(len(self.vertices), 0))

        # O(n)
        for y in range(7,7 + self.meals):
            self.vertices[y].add_edge(self.vertices[y], self.vertices[-1], 0, abs(self.vertices[y].demand), 0) 

        self.vertices[0].add_edge(self.vertices[0], self.vertices[-1], 0, abs(self.vertices[0].demand), 0) 

class ResidualNetwork:
    def __init__(self, graph: Graph) -> None:
        """
        Transforms a graph into a residual network with each edge turned into a current flow and the available
        flow from vertex u to v.

        :Input:
        graph   : a Graph class

        :Output, return or postcondition: A residual network from the input graph with each
                                          edge turned into a flow edge and the remaining flow edge.

        :Time complexity: O(n) where n is the number of days, or len(availability)
        :Aux space complexity: O(n)
        """
        # O(n)
        self.residual = list(Vertex(location_2, graph.vertices[location_2].demand) for location_2 in range(len(graph.vertices)))
        self.meals = graph.meals
        
        # O(n)
        for vertex in graph.vertices:
            # O(n)
            for edge in vertex.edges:
                u = self.residual[edge.u.id]
                v = self.residual[edge.v.id]
                flow = edge.flow
                maximum = edge.maximum
                lower = edge.lower

                if maximum - flow != 0:
                    u.add_edge(u, v, maximum - flow, maximum, lower)
                # If the maximum - flow is 0, then the edge is ignored or available = False.
                else:
                    u.add_edge(u, v, maximum - flow, maximum, lower, False, False)

                if flow != 0:
                    v.add_edge(v, u, flow, maximum, lower, True)
                # If flow is 0 then the edge is ignored or available = False.
                else:
                    v.add_edge(v, u, flow, maximum, lower, True, False)

    def bfs(self, source: Vertex, target: Vertex):
        """
        A bfs algorithm to find a path between the source and the target vertex.

        :Input:
        source  : A vertex which is the start 
        target  : A vertex which is the goal of the bfs

        :Output, return or postcondition: If there is a path between the start and the target,
                                          the target will have a previous, None otherwise.

        :Time complexity: O(n) where n is the number of days, or len(availability)
        :Aux space complexity: O(n)
        """
        discovered = deque([])
        discovered.append(source)
        
        while len(discovered) > 0:
            u = discovered.popleft()
            u.visited = True
            if u == target:
                return

            for edge in u.edges:
                # if edge has a flow of 0, then it is ignored
                if edge.available == False:
                    continue

                v = edge.v

                if v.discovered == False:
                    discovered.append(v)
                    v.discovered = True
                    # tracks the previous edge
                    v.previous = edge
        return 

    def backtrack(self, start: Vertex, finish: Vertex):
        """
        To track the path from the start to the finish vertex using backtracking
        and the self.previous attribute.

        :Input:
        start  : A vertex which is the start of the backtracking
        finish  : A vertex which is the end of the backtracking

        :Output, return or postcondition: Returns a path from the finish to start

        :Time complexity: O(n) where n is the number of days, or len(availability)
        :Aux space complexity: O(n)
        """
        pathh = []
    
        current = start
        
        # O(n) time complexity since at worst case, the loop traverses all vertices
        while current != finish:
        
        # If there is no more previous path in current, return None
        # this means the there is no route between start and stop
            if current == None:
                return pathh

            # Keep traversing the previous vertex as long as it is not None
            pathh.append(current.previous)
            current = current.previous.u
        
        return pathh

    def has_a_path(self, source: Vertex, target: Vertex) -> bool:
        """
        To check if there is a path from source to target.

        :Input:
        source  : A vertex which is the start of the path
        target  : A vertex which is the end of the path

        :Output, return or postcondition: A boolean to determine if there is a path
                                          from source to target.

        :Time complexity: O(n) where n is the number of days, or len(availability)
        :Aux space complexity: O(n)
        """
        self.reset()

        # O(n)
        self.bfs(source, target)

        # If there is no previous of the target after running bfs, then
        # this means that there is no path from source to target
        # O(1)
        if target.previous == None:
            return False
        return True

    def get_minimum(self, path):
        """
        Check the flow of every edge in a path and determine
        the minimum flow.

        :Input:
        path    : a list of edges.

        :Output, return or postcondition: An integer which determine the
                                          the smallest flow among the list of edges.

        :Time complexity: O(n) where n is the number of days, or len(availability)
        :Aux space complexity: O(1)
        """
        minimum = math.inf
        for i in path:
            if i.flow < minimum:
                minimum = i.flow
        return minimum
        
    def get_path(self, source: Vertex, target: Vertex):
        """
        Get the path from source to target using backtracking

        :Input:
        source  : A Vertex which determines the start of the path
        target  : A Vertex which determines the end of the path, if there os any path
                  from the source to target.

        :Output, return or postcondition: A path in a form of list of edges from source to target
                                          using backtracking.

        :Time complexity: O(n) where n is the number of days, or len(availability)
        :Aux space complexity: O(1)
        """
        return self.backtrack(source, target)[::-1]

    def opposite(self, u: Vertex, v: Vertex):
        """
        Get the edge from u to v.

        :Input:
        u: Vertex object
        v: Vertex object

        :Output, return or postcondition: Returns the edge from u to v. Returns None if there
                                          no edge from u to v.

        :Time complexity: O(n) where n is the number of days, or len(availability)
        :Aux space complexity: O(1)
        """
        for edge in u.edges:
            if edge.v == v:
                return edge
        return None

    def update(self, path, change):
        """
        Update the edge between u and v in the residual network.

        :Input:
        path    : a list of edge object
        change  : an integer to determine what to change the edges flow by

        :Output, return or postcondition: A residual network with an updated edge flow 
                                          of the edges in the path.

        :Time complexity: O(n) where n is the number of days, or len(availability)
        :Aux space complexity: O(1)
        """
        for edge in path:
            opposite = self.opposite(edge.v, edge.u)

            edge.flow -= change
            opposite.flow += change

            if edge.flow <= 0:
                edge.available = False
            if opposite.flow > 0:
                opposite.available = True

    def reset(self):
        """
        Resets the attributes of each vertex in the residual network
        so that bfs could be run multiple times.

        :Output, return or postcondition: All the vertices inside the residual network
                                          has its previous, visited, and discovered attributes
                                          reset.

        :Time complexity: O(n) where n is the number of days, or len(availability)
        :Aux space complexity: O(1)
        """
        for vertex in self.residual:
            vertex.previous = None
            vertex.visited = False
            vertex.discovered = False

    def find_previous(self, target):
        """
        Find the previous vertex from the target. This is mainly used to determine
        which person is to cook which meal.

        :Input:
        target: A vertex object which indicates a meal

        :Output, return or postcondition: A vertex id which determines which housemates/ 
                                          restaurant to cook the target meal.

        :Time complexity: O(1)
        :Aux space complexity: O(1)
        """
        # worst case O(1) since at most, the meal will go through
        # 6 edges which are the housemates and the restaurant.
        for edge in target.edges:
            if edge.v.id <= 6:
                return edge.v.id

    def breakfast(self, target):
        """
        To indicate if the target vertex is a breakfast or dinner.

        :Input:
        target: A vertex object which indicates a meal

        :Output, return or postcondition: Returns True if the target vertex is
                                          breakfast meal, returns False if it 
                                          is a dinner meal.

        :Time complexity: O(1)
        :Aux space complexity: O(1)
        """
        if target.id % 2 != 0:
            return True
        return False

    def task(self):
        """
        To check which person will cook each meals or order from the restaurant

        :Output, return or postcondition: Returns a nested list of integers. The first
                                          nested list determines the person who will be cooking breakdast for 
                                          each day. The second nested list determines the person who will be 
                                          cooking dinner for each day. If the integer is 5, then that meal will
                                          be ordered from a restaurant.

        :Time complexity: O(n)
        :Aux space complexity: O(n)
        """
        breakfast = []
        dinner = []

        # O(n)
        for i in range(7, 7 + self.meals):
            # O(n^2)

            for edge in self.residual[i].edges:
                # if edge.v.id > 6, means there is a middle node
                # and we need to call the find_previous node
                if edge.flow == 1 and edge.v.id > 6:

                    if self.breakfast(edge.u):  
                        breakfast.append(self.find_previous(edge.v) - 1)

                    else:
                        dinner.append(self.find_previous(edge.v) - 1)

                elif edge.flow == 1:

                    if self.breakfast(edge.u):
                        breakfast.append(edge.v.id - 1)

                    else:
                        dinner.append(edge.v.id - 1)

        return (breakfast, dinner)

    def possible(self):
        """
        To check if the source output reaches the maximum flow and also,
        the sink state receives the maximum flow. If otherwise, means not possible.

        :Output, return or postcondition: Returns True if the self.residual is possible after
                                          running the Ford-Fulkerson algorithm, False
                                          if otherwise.

        :Time complexity: O(n)
        :Aux space complexity: O(1)
        """
        for edge in self.residual[-2].edges:

            if edge.flow != 0:
                return False

        for edge_2 in self.residual[-1].edges:
            
            if edge_2.flow != edge_2.maximum:
                return False

        return True

def ford_fulkerson(graph: Graph, start: int, finish: int):
    """
    Run Ford-Fulkerson algorithm from start to finish, which is usually
    the source and the sink state. This algorithm is used to distribute 
    the cooking to each housemates or to order from a restaurant. As long as 
    there is still a path from the source to the sink state, the algorithm
    will continue running to find the max flow.

    :Input:
    start   : an integer which is the index of the source vertex
    finish  : an integer which is the index of the sink vertex

    :Output, return or postcondition: Returns a residual network with 
                                      a maximum flow. There will be no more
                                      path from the source to the sink state.

    :Time complexity: O(n^2)
    :Aux space complexity: O(n)
    """
    flow = 0

    # O(n)
    residual_network = ResidualNetwork(graph)

    # loop runs for O(n) times
    while residual_network.has_a_path(residual_network.residual[start], residual_network.residual[finish]):
        # O(n^2)
        path = residual_network.get_path(residual_network.residual[finish], residual_network.residual[start])
        minimum = residual_network.get_minimum(path)
        flow += minimum
        residual_network.update(path, minimum)
    return residual_network

def allocate(availability):
    """
    Distribute the task of preparing meals to 5 housemates and 1 restaurant to
    each day given a constraint of a lower bound and upper bound. 
    This function uses the Ford-Fulkerson algorithm to solve the problem. A person
    cant cook twice in a day and the minimum & maximum requirements must be met.

    :Input:
    availability: a nested list of integers where len(availability) determines the number of days,
                  and each integer in the nested list determines the index's preference of preparing a 
                  meal on that day. If it is 0, the person would like not to make a meal, 1 is only breakfast,
                  2 is only dinner, and 3 if the person doesn't mind doing either one. There are 2 meals
                  in a day which are breakfast and dinner.

    :Output, return or postcondition: Returns a nested list of integers. The first
                                      nested list determines the person who will be cooking breakdast for 
                                      each day. The second nested list determines the person who will be 
                                      cooking dinner for each day. If the integer is 5, then that meal will
                                      be ordered from a restaurant. Returns a None if there is no possible
                                      combination of housemates+restaurant to prepare the meals.

    :Time complexity: O(n^2)
    :Aux space complexity: O(n)
    """

    graph = Graph(availability)
    graph.min_flow()
    graph.optimize()
    residual_network = ford_fulkerson(graph, len(graph.vertices) - 2, len(graph.vertices) - 1)

    if not residual_network.possible():
        return None

    return residual_network.task()
