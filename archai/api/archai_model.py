# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Any, Dict, Optional


class ArchaiModel:
    """Model wrapper with an architecture identifier and an optional metadata dictionary."""

    def __init__(self, arch: Any, archid: str, metadata: Optional[Dict[str, Any]] = None):
        """Initialize the Archai-based model.

        Args:
            arch: Model object (e.g torch.nn.Module).
            archid: String identifier of `arch` object. Will be used to deduplicate models
                of the same architecture, so architecture hashes are prefered. `archid` should
                only identify neural network architectures and not model weight information.
            metadata: Optional model metadata dictionary.

        """

        self.arch = arch
        self.archid = archid
        self.metadata = metadata or {}

    def __repr__(self) -> str:
        return f"ArchaiModel(\n\tarchid={self.archid}, \n\t" f"metadata={self.metadata}, \n\tarch={self.arch}\n)"

    def __str__(self) -> str:
        return repr(self)
