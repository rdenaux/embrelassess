# -*- coding: utf-8 -*-
from unittest import TestCase
import logging
import os
import shutil
import subprocess
import tempfile

logger = logging.getLogger(__name__)


class EmbrelassessTestCase(TestCase):
    def setUp(self):
        logging.basicConfig(format=('%(asctime)s - %(levelname)s - '
                                    '%(name)s - %(message)s'),
                            level=logging.INFO)
        # Directory where everything temporary and test-related is written
        self.project_root = os.path.abspath(os.path.realpath(os.path.join(
            os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir)))
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        try:
            shutil.rmtree(self.test_dir)
            # print('remove test_dir')
        except:
            subprocess.call(["rm", "-rf", self.test_dir])
