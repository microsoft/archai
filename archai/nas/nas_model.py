from typing import Optional, Dict, Any

class NasModel():
    def __init__(self, arch: Any, archid: str, metadata: Optional[Dict] = None):
        """Neural Architecture Search model.

        Args:
            arch (Any): Callable model object
            archid (str): Architecture identifier.
            metadata (Optional[Dict], optional): Extra model metadata. Defaults to None.
        """
        assert isinstance(archid, str)
        assert isinstance(metadata, dict)

        self.arch = arch
        self.archid = archid
        self.metadata = metadata
