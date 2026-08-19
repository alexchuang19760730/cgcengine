# Copyright (c) 2026 SandAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Import every handler module so they register against the global registry."""

from . import arithmetic  # noqa: F401
from . import calls  # noqa: F401
from . import containers  # noqa: F401
from . import control_flow  # noqa: F401
from . import load_store  # noqa: F401
from . import stack_ops  # noqa: F401
