import unittest
from ..models.scrap_book import fun_dict_values


class TestFuncDictValues(unittest.TestCase):

    def test_fun_dict_values(self):
        kwargs_b = {'a': 2, 'b': 3}
        a_values, b_values = fun_dict_values(kwargs_b)
        self.assertEqual(a_values, 4)
        self.assertEqual(b_values, 12)


if __name__ == '__main__':
    unittest.main()