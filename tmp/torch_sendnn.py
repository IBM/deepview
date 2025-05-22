#
# Copyright IBM Corp 2024
#

# This file is left for compatibility purposes and to avoid
# circular dependencies when trying to mix the new style
# backend registration with code that expects to have the backend
# listed in the backends.
# The expectation is that any code which will perform the old-style
# usage will import this submodule. This submodule will force
# resolution of the lazily registered backends (or register if not found)
# allowing for existing code to detect the backends in list_backend.

from torch._dynamo import register_backend
from torch._dynamo.backends.registry import lookup_backend
from torch._dynamo.exc import InvalidBackend

# update_lazyhandle is used by the fms inference server
from torch_sendnn.backends import (
    sendnn_backend,
    sendnn_decoder_backend,
    update_lazyhandle,  # noqa: F401
    preserve_lazyhandle,
    release_lazyhandle
)

_BACKENDS = {
    'sendnn':  sendnn_backend,
    'sendnn_decoder': sendnn_decoder_backend,
}

for name in _BACKENDS:
    compiler_fn = None
    try:
        compiler_fn = lookup_backend(name)
    except InvalidBackend:
        pass

    # This is only necessary if the package was not installed. Lookup
    # failed so we will register manually.
    if compiler_fn is None:
        register_backend(compiler_fn=_BACKENDS[name], name=name)

