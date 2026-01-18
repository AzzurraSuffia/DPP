# label domain class definition 
class LabelDomain():
    def __init__(self, labels, children, root):
        self.labels = labels
        self.root = root
        self.children = children # dict mapping node to children
        self.parent = self._build_parent_map()

    def _build_parent_map(self):
        parent = {self.root: None}
        for p, children in self.children.items():
            for c in children:
                parent[c] = p
        return parent

    def size(self, l):
        if not self.children[l]:
            return 1 # it's a leaf
        
        # Recursive case: sum leaf counts of all children
        return sum(self.size(child_label) for child_label in self.children[l])
    
    def is_leaf(self, l):
        return not self.children[l] # no children

    def normalized_certainty_penalty(self, l): # from 0 (a leaf, max certainty) to 1 (most general label, min certainty)
        root_size = self.size(self.root)
        if root_size <= 1:
            return 0.0  # degenerate domain
        return (self.size(l) - 1) / (root_size - 1)
    
    def find_label_common_parent(self, l1, l2):
        if l1 not in self.labels or l2 not in self.labels:
            return None

        ancestors = []
        current = l1

        # Traverse from l1 up to the root
        while current != '*':
            ancestors.append(current)
            current = self.parent[current]

        ancestors.append('*')

        # Traverse from l2 up to the first common label found
        current = l2
        while current != '*':
            if current in ancestors:
                return current
            current = self.parent[current]

        # If no match is found, return the root
        return '*'