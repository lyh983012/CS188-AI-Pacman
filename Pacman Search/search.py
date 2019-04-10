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

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
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


def get_directions(path):
    """
    :param path: 
    :return: the directions to be taken to follow the path
    """
    return map(lambda x: x[1], path)


def get_current_pos(current_path):
    """
    :param current_path: 
    :return: the (x,y) coordinate of the last position in the path  
    """
    current_state = current_path[-1]
    current_pos = current_state[0]
    return current_pos


def add_successors(fringe, current_path, successors, problem, heuristic):
    """
    :param fringe: holder of the paths yet to be processed
    :param current_path: list of states which make up the path 
    :param successors: future states
    :param problem: not used
    """
    for successor in successors:
        fringe.push(current_path + [successor])


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


def generic_search(fringe, problem, add_successors_fn, heuristic):
    """
    :param fringe: holder of the paths yet to be processed
    :param problem: injected problem
    :param add_successors_fn: defines how the successors are added to the fringe based on the problem
    :param heuristic: function that takes state and problem and returns heuristic cost of choosing that state
    :return: list of directions to be followed to attain the goal defined by problem.isGoalState
    """
    explored = set()
    if problem.isGoalState(problem.getStartState()):
        return []
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
    return []


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    return generic_search(util.Stack(), problem, add_successors, nullHeuristic)


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    return generic_search(util.Queue(), problem, add_successors, nullHeuristic)


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    return generic_search(util.PriorityQueue(), problem, add_successors_with_priority, nullHeuristic)


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    return generic_search(util.PriorityQueue(), problem, add_successors_with_priority, heuristic)


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
