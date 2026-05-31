from abc import ABC, abstractmethod

class BaseKVCache(ABC):
    @abstractmethod
    def write(self, layer_idx, k, v, start_pos):
        pass

    @abstractmethod
    def read(self, layer_idx, upto_pos):
        pass

    @abstractmethod
    def advance(self, amount):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def memory_bytes(self) -> int:
        pass   