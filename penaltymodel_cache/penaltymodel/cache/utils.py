# Copyright 2020 D-Wave Systems Inc.
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
#
# ================================================================================================
from unittest import mock
import tempfile
import os.path as path
from functools import wraps

def isolated_cache(f):
    """Method decorator used to isolate the method's penaltymodel cache."""
    @wraps(f)
    def _isolated_cache(obj):
        with tempfile.TemporaryDirectory() as tmpdir:         
            filename = path.join(tmpdir, "tmp_db_file")

            with mock.patch("penaltymodel.cache.database_manager.cache_file", lambda: filename):
                f(obj)

    return _isolated_cache
    