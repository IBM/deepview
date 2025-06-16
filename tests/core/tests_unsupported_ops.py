# /*******************************************************************************
#  * Copyright 2025 IBM Corporation
#  *
#  * Licensed under the Apache License, Version 2.0 (the "License");
#  * you may not use this file except in compliance with the License.
#  * You may obtain a copy of the License at
#  *
#  *     http://www.apache.org/licenses/LICENSE-2.0
#  *
#  * Unless required by applicable law or agreed to in writing, software
#  * distributed under the License is distributed on an "AS IS" BASIS,
#  * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  * See the License for the specific language governing permissions and
#  * limitations under the License.
# *******************************************************************************/

# Third Party
from sendnn import opcodes
import pytest
import torch

# Local
from deepview.core.unsupported_ops import get_unsupported_ops, sanitize_arg

pytestmark = pytest.mark.skip_debugger_path


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
    """
    Test sanitize_arg with a list input.
    This test checks if the list is converted to a string representation.
    """
    result = sanitize_arg([1, 2, 3])
    print(f"Sanitized arg result: {result}")
    assert result == "[1, 2, 3]"


def test_sanitize_arg_with_float_node():
    """
    Test sanitize_arg with a dtype torch.float node.
    This test checks if the dtype torch.float node is sanitized to a string representation.
    """
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
    """
    Test sanitize_arg with a dtype torch.int32 node.
    This test checks if the dtype torch.int32 node is sanitized to a string representation.
    """
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
    """
    Test sanitize_arg with a dtype torch.bool node.
    This test checks if the dtype torch.bool node is sanitized to a string representation.
    """
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
    """
    Test sanitize_arg with primitive types.
    This test checks if the primitive types are sanitized to their string representations.
    """
    assert sanitize_arg(42) == "42"
    assert sanitize_arg("foo") == "foo"
    assert sanitize_arg(3.14) == "3.14"


""" Unit tests for get_unsupported_ops. """


@pytest.fixture
def DummyLazyHandle():
    """
    A fixture that provides a dummy lazy handle class for testing purposes.
    This class simulates the behavior of a lazy handle that can either have a `g2` attribute
    or a `meta` dictionary containing `g2`. It is used to test the `get_unsupported_ops` function.
    The `DummyLazyHandle` class can be initialized with either a `g2` object or a `meta` dictionary.

    Args:
        g2 (Optional): An optional g2 object to initialize the lazy handle.
        meta (Optional[dict]): An optional dictionary containing a "g2" key to initialize the lazy handle.

    Returns:
        DummyLazyHandle: A dummy lazy handle class that can be used in tests.
    """

    class DummyLazyHandle:
        def __init__(self, g2=None, meta=None):
            # If meta is provided, prefer it; otherwise, use g2 directly.
            if meta is not None:
                self.meta = meta
                # Optionally, set g2 as well for hasattr checks.
                self.g2 = meta.get("g2", None)
            else:
                self.g2 = g2
                self.meta = {"g2": g2} if g2 is not None else {}

    return DummyLazyHandle


@pytest.fixture
def DummyG2():
    """
    A fixture that provides a dummy G2 class for testing purposes.
    This class simulates the behavior of a G2 object that contains a list of compute operations.
    It is used to test the `get_unsupported_ops` function.

    Args:
        compute_ops (list): A list of compute operations to initialize the G2 object.

    Returns:
        DummyG2: A dummy G2 class that can be used in tests.
    """

    class DummyG2:
        def __init__(self, compute_ops):
            self.compute_ops = compute_ops

    return DummyG2


@pytest.fixture
def DummyOp():
    """
    A fixture that provides a dummy operation class for testing purposes.
    This class simulates the behavior of a compute operation with a function and a name.

    Args:
        fn (callable): The function representing the operation based on the torch_sendnn op_codes.
        name (str): The name of the operation.

    Returns:
        DummyOp: A dummy operation class that can be used in tests.
    """

    class DummyOp:
        def __init__(self, fn, name):
            self.fn = fn
            self.name = name

        def Fn(self):
            return self.fn

        def Name(self):
            return self.name

    return DummyOp


def test_get_unsupported_ops_with_g2(DummyOp, DummyLazyHandle, DummyG2):
    """
    Test get_unsupported_ops with a DummyLazyHandle containing a DummyG2 object.
    """
    compute_ops = [
        DummyOp(opcodes.Unsupported, "cos"),
        DummyOp(opcodes.Relu, "relu"),
    ]
    lazy_handle = DummyLazyHandle(g2=DummyG2(compute_ops))
    result = get_unsupported_ops(lazy_handle)
    assert result == ["cos"]


def test_get_unsupported_ops_with_meta(DummyOp, DummyLazyHandle, DummyG2):
    """
    Test get_unsupported_ops with a DummyLazyHandle containing a meta dictionary with g2.
    """
    compute_ops = [
        DummyOp(opcodes.Unsupported, "cos"),
        DummyOp(opcodes.Relu, "relu"),
    ]
    g2 = DummyG2(compute_ops)
    meta = {"g2": g2}
    lh = DummyLazyHandle(meta=meta)
    result = get_unsupported_ops(lh)
    assert result == ["cos"]


def test_get_unsupported_ops_empty(DummyG2, DummyLazyHandle):
    """
    Test get_unsupported_ops with no compute operations object.
    """
    compute_ops = []
    g2 = DummyG2(compute_ops)
    lh = DummyLazyHandle(g2=g2)
    result = get_unsupported_ops(lh)
    assert result == []


def test_get_unsupported_ops_no_unsupported(DummyOp, DummyLazyHandle, DummyG2):
    """
    Test get_unsupported_ops with only supported operations.
    """
    compute_ops = [
        DummyOp(opcodes.Relu, "relu"),
        DummyOp(opcodes.Add, "add"),
    ]
    g2 = DummyG2(compute_ops)
    lh = DummyLazyHandle(g2=g2)
    result = get_unsupported_ops(lh)
    assert result == []
