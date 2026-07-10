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

"""Unit and logging verification tests for time_block ContextManager."""

import logging
from unittest import mock

from absl.testing import absltest
from orbax.checkpoint.experimental.emergency.multi_tier_checkpointing import time_block


class TimeBlockTest(absltest.TestCase):
  """Tests TimeBlock logging and failure behavior."""

  @mock.patch.object(time_block.logging, 'log', autospec=True)
  @mock.patch.object(time_block.time, 'time', side_effect=[10.0, 12.5])
  def test_logs_duration_and_data(self, mock_time, mock_log):
    """Logs appended metrics and elapsed duration at the requested level."""
    del mock_time
    with time_block.TimeBlock('restore', level=logging.DEBUG) as timer:
      timer.append_data('bytes', 1024, 'B', log_rate=True)

    self.assertEqual(timer.duration, 2.5)
    self.assertLen(mock_log.call_args_list, 2)
    mock_log.assert_has_calls([
        mock.call(logging.DEBUG, '+ %s', 'restore', stacklevel=2),
        mock.call(
            logging.DEBUG,
            '%s',
            '- restore; bytes=1.0 KiB @ 409 Bytes/s; duration=2.50 s '
            '(2.50 seconds)',
            stacklevel=2,
        ),
    ])

  @mock.patch.object(time_block.logging, 'log', autospec=True)
  @mock.patch.object(time_block.time, 'time', side_effect=[10.0, 11.0])
  def test_marks_failed_block(self, mock_time, mock_log):
    """Marks the completion log when the wrapped operation raises."""
    del mock_time
    with self.assertRaisesRegex(ValueError, 'failed'):
      with time_block.TimeBlock('save'):
        raise ValueError('failed')

    self.assertTrue(mock_log.call_args_list[-1].args[2].startswith('! save;'))


if __name__ == '__main__':
  absltest.main()
