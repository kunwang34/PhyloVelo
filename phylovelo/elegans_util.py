class Trie:
    '''
    Trie to build phylogenetic tree for C.elegans by it's lineage
    '''
    def __init__(self):
        self.children = {}
        self.isend = False

    def insert(self, word: str) -> None:
        node = self
        for ch in word:
            if not ch in node.children:
                node.children[ch] = Trie()
            node = node.children[ch]
        node.isend = True

    def search(self, word: str) -> bool:
        node = self
        for ch in word:
            if not ch in node.children:
                return False
            node = node.children[ch]
        return node.isend

    def distance(self, word1: str, word2: str) -> int:
        share = 0
        for i, j in zip(word1, word2):
            if i != j:
                break
            share += 1
        return len(word1) + len(word2) - share * 2

    def get_descendant(self, word: str) -> set:
        node = self
        for ind, ch in enumerate(word):
            node = node.children[ch]

        def dfs(prefix, node):
            if node.isend:
                suffix.append(prefix)
            if len(node.children) > 0:
                for i in node.children:
                    dfs(prefix + i, node.children[i])

        prefix = word[: ind + 1]
        suffix = []
        dfs(prefix, node)
        return set(suffix)

    def get_neighbors(self, word: str) -> tuple:
        node = self
        for ind, ch in enumerate(word):
            if len(node.children[ch].children) > 1:
                node1 = node.children[ch]
                ind1 = ind
            node = node.children[ch]

        def dfs(prefix, node):
            if node.isend:
                suffix.append(prefix)
            if len(node.children) > 0:
                for i in node.children:
                    dfs(prefix + i, node.children[i])

        prefix = word[: ind1 + 1]
        suffix = []
        dfs(prefix, node1)
        neighbors = list(set(suffix) - set(self.get_descendant(word))) + [word]
        dist = []
        for i in neighbors:
            dist.append(self.distance(prefix, i))
        neig1, dist1 = [], []
        for i, j in zip(neighbors, dist):
            if j != 0:
                neig1.append(i)
                dist1.append(j)
        return (neig1, dist1, ind1)
