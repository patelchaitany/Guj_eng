class Trie:
    def __init__(self,End):
        self.root = {}
        self.end = End

    def insert(self,word,value):
        node = self.root
        for char in word:
            if char not in node:
                node[char] = {}
            node = node[char]
        node[self.end] = value

    def val_search(self,word):
        node = self.root
        for char in word:
            if char not in node:
                return 0
            node = node[char]
        if self.end not in node:
            return 0
        return node[self.end]

    def set_value(self,word,value):
        node = self.root
        for char in word:
            if char not in node:
                raise ValueError("Word not found")
            node = node[char]
        if self.end not in node:
            raise ValueError("Word not found")
        return node[self.end]
