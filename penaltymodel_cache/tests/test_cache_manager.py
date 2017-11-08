
"""NB: The key features of these tests is that they need to run on all platforms."""
import unittest
import sqlite3
import os

import penaltymodel_cache as pmc


class TestCacheManager(unittest.TestCase):
    def test_cache_file_typical(self):
        """Run cache_file without setting any arguments."""

        filepath = pmc.cache_file()

        # make sure we can open a connection to said file
        conn = sqlite3.connect(filepath)

    def test_cache_database_args(self):
        """Run cache_file with the database fully specified and with
        the directory specified."""

        pth = os.path.join(os.getcwd(), 'tmp/tmp.db')

        filepath = pmc.cache_file(pth)

        # should have done nothing
        self.assertEqual(pth, filepath)

        # run it specifying the directory
        filepath = pmc.cache_file(directory=os.path.join(os.getcwd(), 'tmp'))

        self.assertEqual(os.path.join(os.getcwd(), 'tmp', pmc.cache_manager.DATABASENAME), filepath)
