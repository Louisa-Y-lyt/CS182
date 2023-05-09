# -*- coding: utf-8 -*-
"""
CS 182 Problem Set 1: Python Coding Questions - Fall 2022
Due September 27, 2022 at 11:59pm
"""

### Package Imports ###
import heapq
import abc
from typing import List, Optional, Tuple
### Package Imports ###


# ### Coding Problem Set General Instructions - PLEASE READ ####
# 1. All code should be written in python 3.6 or higher to be compatible with the autograder
# 2. Your submission file must be named "pset1.py" exactly
# 3. No additional outside packages can be referenced or called, they will result in an import error on the autograder
# 4. Function/method/class/attribute names should not be changed from the default starter code provided
# 5. All helper functions and other supporting code should be wholly contained in the default starter code declarations provided.
#    Functions and objects from your submission are imported in the autograder by name, unexpected functions will not be included in the import sequence


class Stack:
    """A container with a last-in-first-out (LIFO) queuing policy."""
    def __init__(self):
        self.list = []

    def push(self,item):
        """Push 'item' onto the stack"""
        self.list.append(item)

    def pop(self):
        """Pop the most recently pushed item from the stack"""
        return self.list.pop()

    def isEmpty(self):
        """Returns true if the stack is empty"""
        return len(self.list) == 0

class Queue:
    """A container with a first-in-first-out (FIFO) queuing policy."""
    def __init__(self):
        self.list = []

    def push(self,item):
        """Enqueue the 'item' into the queue"""
        self.list.insert(0,item)

    def pop(self):
        """
          Dequeue the earliest enqueued item still in the queue. This
          operation removes the item from the queue.
        """
        return self.list.pop()

    def isEmpty(self):
        """Returns true if the queue is empty"""
        return len(self.list) == 0

class PriorityQueue:
    """
      Implements a priority queue data structure. Each inserted item
      has a priority associated with it and the client is usually interested
      in quick retrieval of the lowest-priority item in the queue. This
      data structure allows O(1) access to the lowest-priority item.
    """
    def  __init__(self):
        self.heap = []
        self.count = 0

    def push(self, item, priority):
        entry = (priority, self.count, item)
        heapq.heappush(self.heap, entry)
        self.count += 1

    def pop(self):
        (_, _, item) = heapq.heappop(self.heap)
        return item

    def isEmpty(self):
        return len(self.heap) == 0

    def update(self, item, priority):
        # If item already in priority queue with higher priority, update its priority and rebuild the heap.
        # If item already in priority queue with equal or lower priority, do nothing.
        # If item not in priority queue, do the same thing as self.push.
        for index, (p, c, i) in enumerate(self.heap):
            if i == item:
                if p <= priority:
                    break
                del self.heap[index]
                self.heap.append((priority, c, item))
                heapq.heapify(self.heap)
                break
        else:
            self.push(item, priority)

class SearchProblem(abc.ABC):
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    @abc.abstractmethod
    def getStartState(self) -> "State":
        """
        Returns the start state for the search problem.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def isGoalState(self, state: "State") -> bool:
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def getSuccessors(self, state: "State") -> List[Tuple["State", str, int]]:
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def getCostOfActions(self, actions) -> int:
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        raise NotImplementedError


ACTION_LIST = ["UP", "RIGHT", "DOWN", "LEFT"]


class GridworldSearchProblem(SearchProblem):
    """
    Fill in these methods to define the grid world search as a search problem.
    Actions are of type `str`. Feel free to use any data type/structure to define your states though.
    In the type hints, we use "State" to denote a data structure that keeps track of the state, and you can use
    any implementation of a "State" you want.
    """
    def __init__(self, file, gcost=lambda x: 1):
        """Read the text file and initialize all necessary variables for the search problem"""
        "*** YOUR CODE HERE ***"
        self.file = file
        f = open(self.file, "r")
        text = f.read()
        a = []
        for c in text.split():
            if c != '\n' and c != ' ': 
                a.append(int(c))
        self.size = a[:2]
        map_1 = []
        for r in range(a[0]):
            map_1.append(a[a[1]*r+2:a[1]*(r+1)+2])
        self.mmap = map_1
        self.start = tuple(a[-2:])
        self.start_value = self.mmap[a[-2:][0]][a[-2:][1]]
        self.gcost = gcost
        goal = set()
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                if self.mmap[i][j] == 1: goal.add((i,j))
                
        self.goal = goal
        obst = set()
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                if self.mmap[i][j] == -1: obst.add((i,j))
        self.obst = obst
        self._visited, self._visitedlist = {}, []
        #raise NotImplementedError

    def getStartState(self) -> "State":
        "*** YOUR CODE HERE ***"
        if self.mmap[self.start[0]][self.start[1]] == 1:
            s = set()
            s.add((self.start[0],self.start[1]))
            s = frozenset(s)
        else:
            s = frozenset()
        return (self.start[0],self.start[1],s) 
        raise NotImplementedError

    def directionToVector(self,action):
        diction = [[-1,0],[0,1],[1,0],[0,-1]]
        [dx, dy] = diction[ACTION_LIST.index(action)] 
        return (dx, dy)

    def isGoalState(self, state: "State") -> bool:
        "*** YOUR CODE HERE ***"
        if len(state[2])==len(self.goal):
            return True
        else: return False
        raise NotImplementedError

    def getSuccessors(self, state: "State") -> List[Tuple["State", str, int]]:
        "*** YOUR CODE HERE ***"
        successors = []
        for action in ACTION_LIST:
            x, y = state[0], state[1]
            dx, dy =  self.directionToVector(action)   
            nextx, nexty = int(x + dx), int(y + dy)
            if ((nextx,nexty) not in self.obst and nextx>=0 and nextx< self.size[0] 
            and nexty>=0 and nexty< self.size[1]):
                prev_set = set(state[2])
                if (((nextx,nexty)) in self.goal) and (nextx, nexty) not in state[2]:
                    prev_set.add((nextx, nexty))
                nextState = (nextx, nexty, frozenset(prev_set))
                cost = self.gcost(nextState)
                successors.append((nextState, action, cost))

        return successors
        raise NotImplementedError

    def getCostOfActions(self, actions: List[str]) -> int:
        "*** YOUR CODE HERE ***"
        if actions == None: return 999999
        x, y = self.getStartState()[0],self.getStartState()[1]
        cost = 0
        for action in actions:
            # Check figure out the next state and see whether its' legal
            dx, dy = self.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if (nextx,nexty) in self.obst: return 999999
            cost += self.gcost((x, y))
        return cost
        return NotImplementedError

def add_successors(fringe, current_path, successors, problem, heuristic):
    """
    :param fringe: holder of the paths yet to be processed
    :param current_path: list of states which make up the path 
    :param successors: future states
    :param problem: not used
    """
    for successor in successors:
        fringe.push(current_path + [successor])

def get_current_pos(current_path):
    """
    :param current_path: 
    :return: the (x,y) coordinate of the last position in the path  
    """
    current_state = current_path[-1]
    current_pos = current_state[0]
    return current_pos    

def get_directions(path):
    """
    :param path: 
    :return: the directions to be taken to follow the path
    """
    return map(lambda x: x[1], path)

def generic_search(fringe, problem, add_successors_fn, heuristic):
    """
    :param fringe: holder of the paths yet to be processed
    :param problem: injected problem
    :param add_successors_fn: defines how the successors are added to the fringe based on the problem
    :param heuristic: function that takes state and problem and returns heuristic cost of choosing that state
    :return: list of directions to be followed to attain the goal defined by problem.isGoalState
    """
    fringe = fringe()
    if problem.isGoalState(problem.getStartState()):
        return (1,0) 
    explored = set()

    successors = problem.getSuccessors(problem.getStartState())
    add_successors_fn(fringe, [], successors, problem, heuristic)
    while not fringe.isEmpty():
        current_path = fringe.pop()
        current_pos = get_current_pos(current_path)

        if current_pos not in explored:
            explored.add(current_pos)

            if problem.isGoalState(current_pos):
                return get_directions(current_path)
            successors = problem.getSuccessors(current_pos)
            add_successors_fn(fringe, current_path, successors, problem, heuristic)
    return (0,1)

# +
def depthFirstSearch(problem: SearchProblem) -> List[str]:
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
#     problem_start_ter = list([problem.getStartState()])                          
#     for goal_node in problem.goal:
#         problem_start_ter.append(goal_node)

#     records =  [tuple(problem.getStartState())]
#     solution = []
#     s1 = []
#     while len(records) < len(problem_start_ter):
#         s,c = generic_search(Stack, problem, records[-1], add_successors, nullHeuristic, records)
#         records.append(c)
#         #print(list(s))
#         solution.append(list(s))

#     for i in range(len(solution)):
#         for j in range(len(solution[i])):
#             s1.append(solution[i][j])
#     return s1
    return list(generic_search(Stack, problem, add_successors, nullHeuristic))
    raise NotImplementedError


# +
def breadthFirstSearch(problem: SearchProblem) -> List[str]:
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
#     problem_start_ter = list([problem.getStartState()])                          
#     for goal_node in problem.goal:
#         problem_start_ter.append(goal_node)

#     records =  [tuple(problem.getStartState())]
#     solution = []
#     s1 = []
#     while len(records) < len(problem_start_ter):
#         s, c = generic_search(Queue, problem, records[-1], add_successors, nullHeuristic, records)
#         records.append(c)
#         solution.append(list(s))

#     for i in range(len(solution)):
#         for j in range(len(solution[i])):
#             s1.append(solution[i][j])
    return list(generic_search(Queue, problem, add_successors, nullHeuristic))
    raise NotImplementedError


# -

def nullHeuristic(state: "State", problem: Optional[GridworldSearchProblem] = None) -> int:
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def simpleHeuristic(state: "State", problem: Optional[GridworldSearchProblem] = None) -> int:
    """
    This heuristic returns the number of residences that you have not yet visited.
    """
    return len(problem.goal) - len(state[2])
    raise NotImplementedError

def customHeuristic(state: "State", problem: Optional[GridworldSearchProblem] = None) -> int:
    """
    Create your own heurstic. The heuristic should
        (1) reduce the number of states that we need to search (tested by the autograder by counting the number of
            calls to GridworldSearchProblem.getSuccessors)
        (2) be admissible and consistent
    """
    import math
    max_manhattan_dist = 0
    x = state[0]
    y = state[1] 

    for residence in problem.goal:
        if residence not in set(state[2]):
            max_manhattan_dist = max(abs(state[0] - residence[0]) + abs(state[1] - residence[1]), max_manhattan_dist)
    return max_manhattan_dist
    raise NotImplementedError

def add_successors_with_priority(fringe, current_path, successors, problem, heuristic):
    """
    :param fringe: holder of the paths yet to be processed
    :param current_path: list of states which make up the path 
    :param successors: future states
    :param problem: injected problem
    :param heuristic: function that takes state and problem and returns heuristic cost of choosing that state
    """
    for successor in successors:
        successor_path = current_path + [successor]
        heuristic_cost = heuristic(get_current_pos(successor_path), problem)
        cost_of_action = problem.getCostOfActions(get_directions(successor_path)) + heuristic_cost
        fringe.push(successor_path, cost_of_action)

# +
def aStarSearch(problem: SearchProblem, heuristic=simpleHeuristic) -> List[str]:
    """Search the node that has the lowest combined cost and heuristic first.
    This function takes in an arbitrary heuristic (which itself is a function) as an input."""
    "*** YOUR CODE HERE ***"
#     problem_start_ter = list([problem.getStartState()])                          
#     for goal_node in problem.goal:
#         problem_start_ter.append(goal_node)

#     records =  [tuple(problem.getStartState())]
#     solution = []
#     s1 = []
#     while len(records) < len(problem_start_ter):
#         s, c = generic_search(PriorityQueue, problem, records[-1], add_successors_with_priority, heuristic, records)
#         records.append(c)
#         solution.append(list(s))

#     for i in range(len(solution)):
#         for j in range(len(solution[i])):
#             s1.append(solution[i][j])
    return list(generic_search(PriorityQueue, problem, add_successors_with_priority, heuristic))
    raise NotImplementedError


# -

if __name__ == "__main__":
    ### Sample Test Cases ###
    # Run the following statements below to test the running of your program
    gridworld_search_problem = GridworldSearchProblem("pset1_sample_test_case4.txt") # Test Case 1
    print(depthFirstSearch(gridworld_search_problem))
    print(breadthFirstSearch(gridworld_search_problem))
    print(aStarSearch(gridworld_search_problem))
    
    gridworld_search_problem = GridworldSearchProblem("pset1_sample_test_case2.txt") # Test Case 2
    print(depthFirstSearch(gridworld_search_problem))
    print(breadthFirstSearch(gridworld_search_problem))
    print(aStarSearch(gridworld_search_problem))
    
    gridworld_search_problem = GridworldSearchProblem("pset1_sample_test_case3.txt") # Test Case 3
    print(depthFirstSearch(gridworld_search_problem))
    print(breadthFirstSearch(gridworld_search_problem))
    print(aStarSearch(gridworld_search_problem))


