import collections


class Memory(collections.deque):
    def __init__(self, maxlen: int):
        super(Memory, self).__init__(maxlen=maxlen)
