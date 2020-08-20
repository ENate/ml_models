import unittest
from ..data.make_dataset import process_dataset_func

"""
For the past several weeks I’ve been working on this project on Udemy.
It shows you how you can build a web platform from scratch and make 
it production ready. Having a microservices based backend written in 
Java (with spring boot), a frontend built with Angular, and all of 
these components deployed on and managed by a kubernetes cluster 
that’s hosted on Google Cloud Platform.
I also go through how you can automate your builds and deployments, 
basically full CI/CD pipelines using Gitlab Platform.
"""


class TestMakeDataset(unittest.TestCase):

    def test_process_dataset_func(self):
        self.val_filename = "/foldertodata/data.csv"
        self.assertEqual(process_dataset_func(self.val_filename), 569)

    def test_function_cleveland_data(self, cleveland_filename):
        self.cleveland_filename = "/foldertoanotherdata/"
        # self.assertEqual()

