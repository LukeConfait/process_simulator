import unittest
import nose2

from python_module_and_tests import code


class Test(unittest.TestCase):
    def test1(self):
        self.assertEqual(True, True)


if __name__ == "__main__":
    unittest.main()
