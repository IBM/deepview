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
import torch

# Local
from deepview.core.generate_minimal_repro import sanitize_arg


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
