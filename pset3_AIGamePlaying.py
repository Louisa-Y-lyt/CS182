# -*- coding: utf-8 -*-
"""
CS 182 Problem Set 3: Python Coding Questions - Fall 2022
Due November 7, 2022 at 11:59pm
"""

### Package Imports ###
import abc
import random
from typing import Tuple, List, Type, Optional
### Package Imports ###

#### Coding Problem Set General Instructions - PLEASE READ ####
# 1. All code should be written in python 3.7 or higher to be compatible with the autograder
# 2. Your submission file must be named "pset3.py" exactly
# 3. No additional outside packages can be referenced or called, they will result in an import error on the autograder
# 4. Function/method/class/attribute names should not be changed from the default starter code provided
# 5. All helper functions and other supporting code should be wholly contained in the default starter code declarations provided.
#    Functions and objects from your submission are imported in the autograder by name, unexpected functions will not be included in the import sequence


Action = str


class GameState(abc.ABC):
    """
    This is the abstract class for a game state. Your code should interact with the Ghost game through this interface.
    """

    @abc.abstractmethod
    def is_terminal(self) -> bool:
        """
        Method that returns a boolean value that indicates whether or not the state is a terminal state
        """

    @abc.abstractmethod
    def get_actions(self) -> List[Tuple[Action]]:
        """
        Method that returns a list of children of the current state
        """

    @abc.abstractmethod
    def generate_successor(self, action) -> "GameState":
        """
        Given an action, this method will return the game state that results from taking this action.
        """

    @abc.abstractmethod
    def value(self) -> float:
        """
        Returns the value of the current state if the state is a terminal state
        """


class GhostDictionary:
    """
    DO NOT MODIFY THIS CLASS!
    """
    def __init__(self, dictionary_file):
        self.characters = "abcdefghijklmnopqrstuvwxyz'"
        with open(dictionary_file, "r") as file:
            self.english_words_set = set(file.read().splitlines())

        # create set of all prefixes
        self.all_prefixes = set()
        for word in self.english_words_set:
            prefixes = self.find_prefixes(word)
            for pref in prefixes:
                self.all_prefixes.add(pref)

    # given word, return all of its prefixes
    @staticmethod
    def find_prefixes(word):
        return [word[:i] for i in range(0, len(word) + 1)]


class GhostGameState(GameState):
    """
    Each state holds three pieces of information
    - the current prefix in the game
    - the dictionary being used
    - the index of which player's turn it is

    DO NOT MODIFY THIS CLASS!
    """
    # this variables keeps track of the number of calls to `generate_successor`.
    generate_successor_counter = 0

    def __init__(self, prefix: str, ghost_dictionary: GhostDictionary, index=0):
        self.prefix = prefix
        self.dictionary = ghost_dictionary
        self.index = index

    def get_actions(self) -> List[Action]:
        if self.is_terminal():
            raise Exception("Cannot get the actions from a terminal state")
        legal_actions = []
        for letter in self.dictionary.characters:
            if self.prefix + letter in self.dictionary.all_prefixes:
                legal_actions.append(letter)
        return legal_actions

    def generate_successor(self, action: Action) -> "GameState":
        assert isinstance(action, Action), f"Your action {action} is not valid!"
        assert isinstance(self.prefix, str)
        GhostGameState.generate_successor_counter += 1
        return GhostGameState(self.prefix + action, self.dictionary, (self.index + 1) % 2)

    def is_terminal(self) -> bool:
        return self.prefix in self.dictionary.english_words_set

    def value(self) -> float:
        if not self.is_terminal():
            raise Exception("Not a terminal node")
        return ((-1) ** self.index) / len(self.prefix)


class MultiAgentSearchAgent(abc.ABC):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxAgent, AlphaBetaAgent.

      In the two-player game, each agent is indexed with either 0 or 1.
      The 0 player wants in the final value of the game being as large (positive) as possible.
      The 1 player wants the final value of the game being as small (negative) as possible.
    """

    def __init__(self, index):
        self.index = index

    @abc.abstractmethod
    def get_action(self, game_state: GameState) -> Action:
        """
        Returns the minimax action from the current game_state
        It may be helpful to use the functions `max_val` and `min_val` that you will fill in below,
        """


class MinimaxAgent(MultiAgentSearchAgent):
    def get_action(self, game_state: GameState) -> Action:
        """*** YOUR CODE HERE ***"""
        if game_state.index == 0: 
            return self.max_val(game_state)[1]
        return self.min_val(game_state)[1]

    def max_val(self, game_state: GameState) -> Tuple[float, Optional[Action]]:
        """
        Given a `GameState` object, this function should return a tuple that contains
         - the maximum value that this agent is able to guarantee against any opponent
         - the action necessary to take from the current state corresponding to the maximum value that is returned
            - if `game_state` is already a terminal state, then this should be `None`.
        """
        """*** YOUR CODE HERE ***"""
        if game_state.is_terminal(): return (game_state.value(), None)
        v = float('-inf')
        action = []
        for node in game_state.get_actions():
            next_game_state = game_state.generate_successor(node)
            res = self.min_val(next_game_state)
            if res[0] > v: 
                v = res[0]
                action = node
        return (v, action)

    def min_val(self, game_state: GameState) -> Tuple[float, Optional[Action]]:
        """
        Given a `GameState` object, this function should return a tuple that contains
         - the minimum value that this agent is able to guarantee against any opponent
         - the action necessary to take from the current state corresponding to the minimum value that is returned
            - if `game_state` is already a terminal state, then this should be `None`.
        """
        "*** YOUR CODE HERE ***"
        """*** YOUR CODE HERE ***"""
        if game_state.is_terminal(): return (game_state.value(), None)
        v = float('inf')
        action = []
        for node in game_state.get_actions():
            next_game_state = game_state.generate_successor(node)
            res = self.max_val(next_game_state)
            if res[0] < v: 
                v = res[0]
                action = node
        return (v, action)


class AlphaBetaAgent(MultiAgentSearchAgent):
    
    def get_action(self, game_state: GameState):
        """*** YOUR CODE HERE ***"""
        if game_state.index == 0: 
            return self.max_val(game_state, -1, 1)[1]
        return self.min_val(game_state, -1, 1)[1]

    def max_val(self, game_state: GameState, alpha: float, beta: float) -> Tuple[float, Optional[Action]]:
        """*** YOUR CODE HERE ***"""
        if game_state.is_terminal(): return (game_state.value(), None)
        v = -1
        for node in game_state.get_actions():
            next_game_state = game_state.generate_successor(node)
            res = self.min_val(next_game_state, alpha, beta)
            if res[0] > v: 
                v = res[0]
                result = node
            if v >= beta: return (v, result)
            if v >= alpha: alpha = v
        return (v, result)

    def min_val(self, game_state: GameState, alpha: float, beta: float) -> Tuple[float, Optional[Action]]:
        """*** YOUR CODE HERE ***"""
        if game_state.is_terminal(): return (game_state.value(), None)
        v = +1
        for node in game_state.get_actions():
            next_game_state = game_state.generate_successor(node)
            res = self.max_val(next_game_state, alpha, beta)
            if res[0] < v: 
                v = res[0]
                result = node
            if v <= alpha: return (v, result)
            if v <= beta: beta = v
        return (v, result)


class RandomAgent(MultiAgentSearchAgent):
    def get_action(self, game_state: GameState) -> Action:
        return random.choice(game_state.get_actions())


class OptimizedAgainstRandomAgent(MultiAgentSearchAgent):

    def get_action(self, game_state: GameState) -> Action:
        """*** YOUR CODE HERE ***"""
        if game_state.index == 0: 
            return self.max_val(game_state)[1]
        return self.min_val(game_state)[1]
    
    def max_val(self, game_state: GameState) -> Tuple[float, Optional[Action]]:
        """*** YOUR CODE HERE ***"""
        if game_state.is_terminal(): return (game_state.value(), None)
        v = float('-inf')
        action = None
        for node in game_state.get_actions():
            next_game_state = game_state.generate_successor(node)
            res = self.get_expected_value(next_game_state)
            if res > v:
                v = res
                action = node
        return (v, action)
    
    def min_val(self, game_state: GameState) -> Tuple[float, Optional[Action]]:
        """*** YOUR CODE HERE ***"""
        if game_state.is_terminal(): return (game_state.value(), None)
        v = float('inf')
        action = None
        for node in game_state.get_actions():
            next_game_state = game_state.generate_successor(node)
            res = self.get_expected_value(next_game_state)
            if res < v: 
                v = res
                action = node
        return (v, action) 

    def get_expected_value(self, game_state: GameState) -> float:
        if game_state.is_terminal(): return game_state.value()
        total = 0
        for act in game_state.get_actions():
            s = game_state.generate_successor(act)
            total += self.get_expected_value(s)
        return total / len(game_state.get_actions())


def play_game(
        dictionary: GhostDictionary,
        starting_prefix: str,
        starting_player: int,
        agent_class_0: Type[MultiAgentSearchAgent],
        agent_class_1: Type[MultiAgentSearchAgent],
        verbose: bool = True
) -> float:
    """
    This is a helper function for you to simulate the games locally.

    Given a GhostDictionary, a starting prefix, and the index of the starting player, and the class of the agent,
    this function simulates the full gameplay and returns the terminal state's value.
    """
    GhostGameState.generate_successor_counter = 0   # reset the counter for every game!
    start_state = GhostGameState(starting_prefix, dictionary, starting_player)
    if verbose:
        print(f"Starting prefix: {starting_prefix}. Starting agent: {starting_player}")
    assert not start_state.is_terminal(), "Prefix input is already a word! This is invalid."
    assert starting_prefix in dictionary.all_prefixes, "Prefix input is not in the set of valid prefixes!"
    state = start_state
    agents = [agent_class_0(0), agent_class_1(1)]
    current_index = starting_player
    while True:
        action = agents[current_index].get_action(state)
        state = state.generate_successor(action)
        if verbose:
            print(f"Agent {current_index} placed a {action}, bringing the current prefix to {state.prefix}")
        if state.is_terminal():
            if verbose:
                print(f"Total number of calls to `generate_successor`: {state.generate_successor_counter}")
                print("The game is over! Value: ", state.value())
            return state.value()
        current_index = (current_index + 1) % 2


def simulate_versus_random(dictionary: GhostDictionary, prefix: str, k: int = 10000) -> Tuple[float, float]:
    """This is a helper function for you to answer part (5)."""
    optimal_vs_random_value = 0
    minimax_vs_random_value = 0
    for _ in range(k):
        optimal_vs_random_value += play_game(dictionary, prefix, 0, OptimizedAgainstRandomAgent, RandomAgent, False)
        minimax_vs_random_value += play_game(dictionary, prefix, 0, MinimaxAgent, RandomAgent, False)
    optimal_vs_random_value /= k
    minimax_vs_random_value /= k
    return optimal_vs_random_value, minimax_vs_random_value


if __name__ == "__main__":
    dictionary = GhostDictionary("dictionary.txt")
    prefix = "enf"
    play_game(dictionary, prefix, 0, MinimaxAgent, MinimaxAgent)
