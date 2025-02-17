###################### defining log trasformed interval ######################
import math
from typing import Any, Dict, List, Optional, Union

import torch
from botorch.exceptions import UnsupportedError
from botorch.fit import fit_gpytorch_mll
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform
from gpytorch import settings
from gpytorch.constraints import Interval
from torch import Tensor
from torch.nn import Parameter


class LogTransformedInterval(Interval):
    """Modification of the GPyTorch interval class.

    The Interval class in GPyTorch will map the parameter to the range [0, 1] before
    applying the inverse transform. We don't want to do this when using log as an
    inverse transform. This class will skip this step and apply the log transform
    directly to the parameter values so we can optimize log(parameter) under the bound
    constraints log(lower) <= log(parameter) <= log(upper).
    """

    def __init__(self, lower_bound, upper_bound, initial_value=None):
        super().__init__(
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            transform=torch.exp,
            inv_transform=torch.log,
            initial_value=initial_value,
        )

        # Save the untransformed initial value
        self.register_buffer(
            "initial_value_untransformed",
            (
                torch.tensor(initial_value).to(self.lower_bound)
                if initial_value is not None
                else None
            ),
        )

        if settings.debug.on():
            max_bound = torch.max(self.upper_bound)
            min_bound = torch.min(self.lower_bound)
            if max_bound == math.inf or min_bound == -math.inf:
                raise RuntimeError(
                    "Cannot make an Interval directly with non-finite bounds. Use a derived class like "
                    "GreaterThan or LessThan instead."
                )

    def transform(self, tensor):
        if not self.enforced:
            return tensor

        transformed_tensor = self._transform(tensor)
        return transformed_tensor

    def inverse_transform(self, transformed_tensor):
        if not self.enforced:
            return transformed_tensor

        tensor = self._inv_transform(transformed_tensor)
        return tensor
