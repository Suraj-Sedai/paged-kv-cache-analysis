from abc import ABC, abstractmethod

class BaseKVCache(ABC):
    @abstractmethod
    def write(self, layer_idx, seq_id, k, v):
        pass

    @abstractmethod
    def read(self, layer_idx, seq_id):
        pass

    @abstractmethod
    def advance(self, seq_id, amount=1):
        pass

    @abstractmethod
    def free(self, seq_id):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def memory_bytes(self) -> int:
        pass

    @abstractmethod
    def fragmentation_ratio(self, seq_id) -> float:
        pass