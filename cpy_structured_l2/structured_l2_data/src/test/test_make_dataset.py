import unittest
from ..data.make_dataset import process_dataset_func

"""
"""


class TestMakeDataset(unittest.TestCase):

    def test_process_dataset_func(self):
<<<<<<< HEAD
        self.val_filename = "/foldertodata/data.csv"
        self.assertEqual(process_dataset_func(self.val_filename), 569)

    def test_function_cleveland_data(self, cleveland_filename):
        self.cleveland_filename = "/foldertoanotherdata/"
=======
        self.val_filename = "pathtodata/data.csv"
        self.assertEqual(process_dataset_func(self.val_filename), 569)

    def test_function_cleveland_data(self, cleveland_filename):
        self.cleveland_filename = "pathtoanotherdata/data_cleveland/"
>>>>>>> 86d38279157a0b87dc04373d2702c39f789e76fd
        # self.assertEqual()

