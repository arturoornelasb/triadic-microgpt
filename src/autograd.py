"""
Autograd Engine — Scalar-level automatic differentiation.

Inspired by Andrej Karpathy's microgpt.py. Enhanced with tanh and
additional utilities needed for the triadic projection head.

Every Value node tracks its data, gradient, children, and local gradients.
Calling .backward() on the loss triggers reverse-mode autodiff through
the entire computation graph.
"""

import math


class Value:
    """
    A scalar value that tracks its computation graph for automatic differentiation.

    Supports: +, -, *, /, **, log, exp, relu, tanh, and backward().

    Usage:
        a = Value(2.0)
        b = Value(3.0)
        c = a * b + a ** 2
        c.backward()
        print(a.grad)  # dc/da
    """

    __slots__ = ('data', 'grad', '_children', '_local_grads')

    def __init__(self, data, children=(), local_grads=()):
        self.data = data
        self.grad = 0
        self._children = children
        self._local_grads = local_grads

    # --- Arithmetic operations ---

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, other):
        return Value(
            self.data ** other,
            (self,),
            (other * self.data ** (other - 1),)
        )

    def log(self):
        return Value(math.log(self.data), (self,), (1 / self.data,))

    def exp(self):
        e = math.exp(self.data)
        return Value(e, (self,), (e,))

    def relu(self):
        return Value(max(0, self.data), (self,), (float(self.data > 0),))

    def tanh(self):
        t = math.tanh(self.data)
        return Value(t, (self,), (1 - t * t,))

    # --- Reverse operations ---

    def __neg__(self):
        return self * -1

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * other ** -1

    def __rtruediv__(self, other):
        return other * self ** -1

    # --- Backward pass ---

    def backward(self):
        """
        Reverse-mode automatic differentiation.
        Computes gradients for all nodes in the computation graph
        by topologically sorting the graph and applying the chain rule.
        """
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        self.grad = 1

        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * v.grad

    # --- Utilities ---

    def __repr__(self):
        return f"Value(data={self.data:.6f}, grad={self.grad:.6f})"

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other
