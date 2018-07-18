class FPNode:
    """ node in Fp tree"""

    def __init__(self, tree, item, count = 1):
        self._tree = tree
        self._item = item
        self._count = count
        self._parent = None
        self._children = {}
        self._neighbor = None

    def add(self, child):
        """ given a FPNode as a child of this node"""

        # chech child's class type
        if not isinstance(child, FPNode):
            raise TypeError('can only add other FPNodes as child')

        if not child.item in self._children:
            self._children[child.item] = child
            child.parent = self

    def search(self, item):
        """ Check whether this node contains a child node for the given item.
        If so, that node is returned; otherwise, `None` is returned."""
        try:
            return self._children[item]
        except KeyError:
            return None

    @property
    def tree(self):
        return self._tree

    @property
    def item(self):
        return self._item

    def increment(self):
        if self._count is None:
            raise ValueError('Root has no associated value')
        self._count += 1

    @property
    def root(self):
        return self._item is None and self._count is None

    @property
    def leaf(self):
        return len(self._children) == 0

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, value):
        if value is not None and not isinstance(value, FPNode):
            raise TypeError("A node must have an FPNode as a parent.")
        if value and value.tree is not self.tree:
            raise ValueError("Cannot have a parent from another tree.")
        self._parent = value

    @property
    def neighbor(self):
        return self._neighbor

    @neighbor.setter
    def neighbor(self, value):
        if value is not None and not isinstance(value, FPNode):
            raise TypeError("A node must have an FPNode as a neighbor.")
        if value and value.tree is not self.tree:
            raise ValueError("Cannot have a neighbor from another tree.")
        self._neighbor = value

    @property
    def children(self):
        return tuple(self._children)

    def inspect(self, depth=0):
        print('  ' * depth) + repr(self)
        for child in self.children:
            child.inspect(depth + 1)

    @property
    def count(self):
        """The count associated with this node's item."""
        return self._count
