# Copyright 2026 The Orbax Authors.
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

from absl import flags
from orbax.checkpoint._src.checkpointers import checkpointer as checkpointer_lib
from orbax.checkpoint._src.checkpointers import checkpointer_test_utils
from orbax.checkpoint._src.testing import multiprocess_test


FLAGS = flags.FLAGS


class CheckpointerTest(
    checkpointer_test_utils.CheckpointerTestBase.Test,
    multiprocess_test.MultiProcessTest,
):

  def checkpointer(self, handler, **kwargs):
    return checkpointer_lib.Checkpointer(handler, **kwargs)


if __name__ == '__main__':
  multiprocess_test.main()
