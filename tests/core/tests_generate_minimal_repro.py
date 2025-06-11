import torch
from deepview.core.generate_minimal_repro import (
    sanitize_arg,
)


class DummyNode(torch.fx.Node):
    """
    A dummy node class to simulate a torch.fx.Node with additional attributes.

    Args:
        torch.fx.Node: Inherits from torch.fx.Node.
    """

    def __init__(
        self, graph, name, op, target, args, kwargs, dtype, shape, type_expr=None
    ):
        """
        Initializes the DummyNode with the given parameters.

        Args:
            graph (torch.fx.Graph): The graph this node belongs to.
            name (str): Name of the node.
            op (str): Operation type of the node.
            target (callable): The target function or operation.
            args (tuple): Positional arguments for the operation.
            kwargs (dict): Keyword arguments for the operation.
            dtype (torch.dtype): Data type of the tensor.
            shape (tuple): Shape of the tensor.
            type_expr: Optional type expression for the node.
        """
        super().__init__(graph, name, op, target, args, kwargs, type_expr)
        self.dtype = dtype
        self.shape = shape


graph = torch.fx.Graph()

""" Unit tests for sanitize_arg. """

def test_sanitize_arg_with_list():
    result = sanitize_arg([1, 2, 3])
    print(f"Sanitized arg result: {result}")
    assert result == "[1, 2, 3]"


def test_sanitize_arg_with_float_node():
    node = DummyNode(
        graph=graph,
        name="dummy_node",
        op="call_function",
        target=torch.add,
        args=(1, 2),
        kwargs={},
        dtype=torch.float32,
        shape=(1, 8, 64),
    )
    result = sanitize_arg(node)
    assert "torch.rand" in result


def test_sanitize_arg_with_int_node():
    node = DummyNode(
        graph=graph,
        name="dummy_node",
        op="call_function",
        target=torch.add,
        args=(1, 2),
        kwargs={},
        dtype=torch.int32,
        shape=(4,),
    )
    result = sanitize_arg(node)
    assert "torch.randint" in result


def test_sanitize_arg_with_bool_node():
    node = DummyNode(
        graph=graph,
        name="dummy_node",
        op="call_function",
        target=torch.add,
        args=(1, 2),
        kwargs={},
        dtype=torch.bool,
        shape=(1, 2),
    )
    result = sanitize_arg(node)
    assert "torch.rand" in result and "< 0.9" in result


def test_sanitize_arg_with_primitive():
    assert sanitize_arg(42) == "42"
    assert sanitize_arg("foo") == "foo"
    assert sanitize_arg(3.14) == "3.14"
