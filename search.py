# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def expand(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (child,
        action, stepCost), where 'child' is a child to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that child.
        """
        util.raiseNotDefined()

    def getActions(self, state):
        """
          state: Search state

        For a given state, this should return a list of possible actions.
        """
        util.raiseNotDefined()

    def getActionCost(self, state, action, next_state):
        """
          state: Search state
          action: action taken at state.
          next_state: next Search state after taking action.

        For a given state, this should return the cost of the (s, a, s') transition.
        """
        util.raiseNotDefined()

    def getNextState(self, state, action):
        """
          state: Search state
          action: action taken at state

        For a given state, this should return the next state after taking action from state.
        """
        util.raiseNotDefined()

    def getCostOfActionSequence(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    """

    # stack to keep track of where our agent can move
    frontier = util.Stack()
    # start node contains the start state and an empty list representing the actions its taken thusfar
    startNode = (problem.getStartState(), [])
    frontier.push(startNode)

    # holds the states visited so we dont end up in an infinite loop
    expanded = []

    while not frontier.isEmpty():
        node = frontier.pop()
        # the current state of our node
        currentState = node[0]
        # the path we took to get to our current state
        actionPath = node[1]

        # if we are at the goal, then return path to the goal
        if problem.isGoalState(currentState):
            return actionPath
        # ensure that we are not double visiting a node
        if currentState not in expanded:
            # add the current state to our expanded so we dont go over it more than once
            expanded.append(currentState)
            # children is a list of nodes connected to our currentState in format (child, action, stepCost)
            children = problem.expand(currentState)
            for child in children:
                # throw node on stack in form new state, newPath
                childNode = (child[0], actionPath + [child[1]])
                frontier.push(childNode)
    # if we get through the entire search frontier without reaching the goal, return None
    return None


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    # Same as DFS except using the Queue instead of the stack
    # Queue to keep track of where our agent can move
    frontier = util.Queue()
    # start node contains the start state and an empty list representing the actions its taken thusfar
    startNode = (problem.getStartState(), [])
    frontier.push(startNode)

    # holds the states visited so we dont end up in an infinite loop
    expanded = []

    while not frontier.isEmpty():
        node = frontier.pop()
        # the current state of our node
        currentState = node[0]
        # the path we took to get to our current state
        actionPath = node[1]

        # if we are at the goal, then return path to the goal
        if problem.isGoalState(currentState):
            return actionPath
        # ensure that we are not double visiting a node
        if currentState not in expanded:
            # add the current state to our expanded so we dont go over it more than once
            expanded.append(currentState)
            # children is a list of nodes connected to our currentState in format (child, action, stepCost)
            children = problem.expand(currentState)
            for child in children:
                # throw node on stack in form new state, newPath
                childNode = (child[0], actionPath + [child[1]])
                frontier.push(childNode)
    # if we get through the entire search frontier without reaching the goal, return None
    return None


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    # priority queue to keep track of where our agent can move
    frontier = util.PriorityQueue()
    # start node contains the start state, an empty list representing the actions its taken thusfar
    startNode = (problem.getStartState(), [])
    # push node onto the stack with a prio of 0 (arbitrary bc start)
    frontier.push(startNode, 0)

    # holds the states visited so we dont end up in an infinite loop
    expanded = []

    while not frontier.isEmpty():
        node = frontier.pop()
        # the current state of our node
        currentState = node[0]
        # the path we took to get to our current state
        actionPath = node[1]
        # if we are at the goal, then return path to the goal
        if problem.isGoalState(currentState):
            return actionPath
        # ensure that we are not double visiting a node
        if currentState not in expanded:
            # add the current state to our expanded so we dont go over it more than once
            expanded.append(currentState)
            # children is a list of nodes connected to our currentState in format (child, action, stepCost)
            children = problem.expand(currentState)
            for child in children:
                # throw node on stack in form new state, newPath, stepCost
                newPath = actionPath + [child[1]]
                childState = child[0]
                childNode = (childState, newPath)

                # g(n) -- the observed cost from startState to the child state
                cost = problem.getCostOfActionSequence(newPath)
                # g(n) + h(n) -- take g(n) and add h(n) which is the heuristic estimate of child state to goal
                cost += heuristic(childState, problem)

                frontier.push(childNode, cost)
    # if we get through the entire search frontier without reaching the goal, return None
    return None

def iterativeDeepeningSearch(problem):
    '''Implements iterative deepening search to find an optimal solution
    path for the given problem.

    This is done by repeatedly running a depth-limited version of
    depth-first search for increasing depths.  A depth-limited search
    only expands a node--retrieves its children and adds them to the
    frontier--if the path to the node has a number of actions that is
    no more than the given depth.  If a node would have been expanded
    except for the depth limit, we say that the search has been "cut
    off" for that depth limit.  As suggested by the slides, we will
    run this search for depth limits of 1, 2, 3, etc., until either a
    solution is found or a depth-limited search has been run without
    being cut off for a given depth limit and without finding a
    solution, in which case no goal node can be reached and
    iterativeDeepeningSearch should return None.

    Args:
      problem: A SearchProblem instance that defines a search problem.

    Returns:
      A list of actions that is as short as possible for reaching a
        goal node, or None if no goal node is reachable from the initial
        state.

    '''
    depth_limit = 0
    while True:
        depth_limit += 1
        # print(f'Search to depth {depth_limit}')
        result = limitedDepthFirstSearch(problem, depth_limit)
        if result != "cutoff":
            return result


def limitedDepthFirstSearch(problem, depth_limit=1):
    '''Runs depth-first search with a depth bound.

    Args:
      problem: A SearchProblem instance that defines a search problem.
      depth_limit: A node will not be expanded if the path to the node
        has a length exceeding the depth value, which is expected to
        be a positive integer.

    Returns:
      A path to a goal node, if one is found, or the string "cutoff" if no
      goal was found and the search was cut off for the given depth limit,
      or None if no goal was found and the search was not cut off.
    '''

    class Node:
        def __init__(self, state, path):
            self.state = state
            self.path = path

    def isCycle(node, expanded):
        for expand in expanded:
            # if (node.state == expand.state) and (node.path in expand.path):
            if node.state == expand.state:
                return True
        return False

    node = Node(problem.getStartState(), ())
    frontier = util.Stack()
    frontier.push(node)
    expanded = []
    result = None
    while not frontier.isEmpty():
        node = frontier.pop()
        if problem.isGoalState(node.state):
            return list(node.path)
        if len(node.path) > depth_limit:
            result = "cutoff"
        elif not isCycle(node, expanded):
        #elif node.state not in expanded:
            expanded.append(node)
            child_nodes = problem.expand(node.state)
            for child in child_nodes:
                frontier.push(Node(child[0], node.path + (child[1],)))
    return result





# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
# Abbreviation for use on command line
ids = iterativeDeepeningSearch
