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

from unittest import mock
from absl import flags
from absl.testing import absltest
from etils import epath
from orbax.checkpoint._src.testing.benchmarks.safetensors import prepare


class PrepareTest(absltest.TestCase):

  @mock.patch.object(prepare, "_snapshot_download")
  @mock.patch("subprocess.run")
  def test_main_with_repo_and_gcs(self, mock_run, mock_download):
    flags.FLAGS.repo = "dummy-org/dummy-repo"
    flags.FLAGS.gcs = "gs://dummy-bucket/model"
    flags.FLAGS.local_dir = None

    temp_dir = self.create_tempdir()
    (epath.Path(temp_dir.full_path) / "file1.safetensors").write_text("dummy")
    (epath.Path(temp_dir.full_path) / "file2.safetensors").write_text("dummy")

    mock_download.return_value = epath.Path(temp_dir.full_path)

    prepare.main([])

    mock_download.assert_called_once_with("dummy-org/dummy-repo", mock.ANY)
    mock_run.assert_called_with(
        [
            "gcloud",
            "storage",
            "rsync",
            temp_dir.full_path,
            "gs://dummy-bucket/model",
            "-r",
        ],
        check=True,
    )



if __name__ == "__main__":
  # Bypass required flag check during test initialization
  flags.FLAGS.set_default("repo", "dummy-org/dummy-repo")
  absltest.main()
