from typing import Any, Optional

from torch import nn, Tensor, flatten
import torch
from torch.ao.quantization import DeQuantStub, QuantStub

from model import CustomInvertedResidual, CustomMobileNetV2

from torchvision.ops import Conv2dNormActivation
from torchvision.models.quantization.utils import _fuse_modules


class CustomQuantizableInvertedResidual(CustomInvertedResidual):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x: Tensor) -> Tensor:
        if self.use_res_connect:
            return self.skip_add.add(x, self.conv(x))
        else:
            return self.conv(x)

    def fuse_model(self, is_qat: Optional[bool] = None) -> None:
        for idx in range(len(self.conv)):
            if type(self.conv[idx]) is nn.Conv2d:
                _fuse_modules(self.conv, [str(idx), str(idx + 1)], is_qat, inplace=True)


class CustomQuantizableMobileNetV2(CustomMobileNetV2):
    def __init__(self, num_skip_qat_layers: int = 0, *args: Any, **kwargs: Any) -> None:
        """
        MobileNet V2 main class

        Args:
           Inherits args from floating point MobileNetV2
        """
        super().__init__(*args, **kwargs)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

        assert len(self.features) >= num_skip_qat_layers
        self.num_skip_qat_layers = num_skip_qat_layers

    def forward(self, x: Tensor) -> Tensor:
        x = self.quant(x)

        # Split features for qat and non-qat
        qat_features = self.features if self.num_skip_qat_layers == 0 else self.features[:-1 * self.num_skip_qat_layers]
        non_qat_features = self.features[len(self.features) - self.num_skip_qat_layers:]

        for f in qat_features:
            x = f(x)

        x = self.dequant(x)

        # Pass data through features not to be quantized
        for f in non_qat_features:
            x = f(x)

        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = flatten(x, 1)
        x = self.classifier(x)
        return x

    def setup_qconfig(self, engine: str) -> None:
        # Disable quantization config for layers that are NOT to be quantized
        torch.backends.quantized.engine = engine
        self.qconfig = torch.quantization.get_default_qat_qconfig(engine)
        self.classifier.qconfig = None
        for f in self.features[len(self.features) - self.num_skip_qat_layers:]:
            f.qconfig = None

    def fuse_model(self, is_qat: Optional[bool] = None) -> None:
        for m in self.modules():
            if type(m) is Conv2dNormActivation:
                _fuse_modules(m, ["0", "1", "2"], is_qat, inplace=True)
            if type(m) is CustomQuantizableInvertedResidual:
                m.fuse_model(is_qat)
