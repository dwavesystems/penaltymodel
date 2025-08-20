# Copyright 2021 D-Wave Systems Inc.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from warnings import warn as _warn  # so it doesn't get pulled in by import *

__version__ = '1.2.0'

import penaltymodel.core

from penaltymodel.database import *
import penaltymodel.database

from penaltymodel.exceptions import *
import penaltymodel.exceptions

from penaltymodel.interface import *
import penaltymodel.interface

from penaltymodel.typing import *
import penaltymodel.typing

from penaltymodel.utils import *
import penaltymodel.utils

_warn("penaltymodel is deprecated and will be removed in Ocean 10. "
      "For solving problems with constraints, "
      "we recommend using the hybrid solvers in the Leap service. "
      "You can find documentation for the hybrid solvers at https://docs.dwavequantum.com.",
      category=DeprecationWarning,
      stacklevel=2,
)
