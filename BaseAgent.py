from abc import ABC, abstractmethod


class BaseAgent(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def take_action(self, state):
        raise NotImplementedError

    @abstractmethod
    def update(self, transition_dict):
        raise NotImplementedError
