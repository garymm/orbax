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

"""Context manager for logging the duration and metrics of an operation."""

import datetime as dt
import logging
import time
from typing import Any

import humanize
import humanize.filesize


class TimeBlock:
  """Logs an operation's start, completion, duration, and optional metrics."""

  def __init__(self, msg: str, level: int = logging.INFO) -> None:
    self.msg = msg
    self.level = level
    self.data: dict[str, tuple[Any, str, bool]] = {}
    self.start_time = 0.0
    self.duration = 0.0

  def __enter__(self):
    logging.log(self.level, '+ %s', self.msg, stacklevel=2)
    self.start_time = time.time()
    return self

  def __exit__(self, exc_type, exc_value, traceback) -> None:
    del exc_value, traceback
    self.duration = time.time() - self.start_time
    msg = '! ' if exc_type else '- '
    msg += self.msg
    self.append_data('duration', self.duration, 's')
    for key, (value, unit, log_rate) in self.data.items():
      val = self._format_value(value, unit, log_rate)
      msg += f'; {key}={val}'

    logging.log(self.level, '%s', msg, stacklevel=2)

  def _format_value(self, value: Any, unit: str, log_rate: bool) -> str:
    """Formats a metric value and its optional per-second rate."""
    if isinstance(value, (int, float)):
      if self.duration == 0:
        log_rate = False

      # Special cases first.
      if unit == 's':
        hum = humanize.precisedelta(
            dt.timedelta(seconds=value), format='%0.2f'
        )
        hum = f'{value:.2f} s ({hum})'
      elif unit == 'B':
        hum = humanize.filesize.naturalsize(value, binary=True)
        if log_rate:
          rate = humanize.filesize.naturalsize(
              value / self.duration, binary=True
          )
          hum += f' @ {rate}/s'
      else:
        hum = humanize.intcomma(value)
        if unit:
          hum += ' ' + unit
        if log_rate:
          rate = humanize.intcomma(value / self.duration, 3)
          rate_unit = f' {unit}/s' if unit else ' / s'
          hum += f' @ {rate}{rate_unit}'
    else:
      hum = str(value)

    return hum

  def append_msg(self, msg: str) -> None:
    """Appends text to the operation description."""
    self.msg += f'; {msg}'

  def append_data(
      self,
      key: str,
      value: Any,
      unit: str = '',
      log_rate: bool = False,
  ) -> None:
    """Adds a metric to the completion log."""
    self.data[key] = (value, unit, log_rate)


if __name__ == '__main__':
  logging.basicConfig(
      level=logging.INFO,
      format='%(asctime)s: %(message)s',
      datefmt='%Y-%m-%d %H:%M:%S',
  )
  with TimeBlock('Hello'):
    time.sleep(1)
