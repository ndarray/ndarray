import ndarray_tables_mod
import unittest
import numpy

class TestWrappers(unittest.TestCase):

    def testLayout(self):
        l1 = ndarray_tables_mod.makeLayout(3, 2, 5, True)
        l2 = numpy.dtype([("a", numpy.int32), ("b", numpy.float64, (3,2)), ("c", numpy.float32, (5,))])
        self.assertEqual(l1, l2)
        self.assert_(ndarray_tables_mod.compareLayouts(l2, 3, 2, 5, True))

    def testTable(self):
        l1 = ndarray_tables_mod.makeLayout(3, 2, 5, False)
        t1 = ndarray_tables_mod.makeTable(4, l1)
        self.assertEqual(l1, t1.dtype)
        t1["a"] = numpy.random.randint(low=0, high=100, size=t1["a"].shape)
        t1["b"] = numpy.random.randn(*t1["b"].shape)
        t1["c"] = numpy.random.randn(*t1["c"].shape)
        self.assert_(ndarray_tables_mod.compareTables(t1, t1["a"], t1["b"], t1["c"]))

if __name__=="__main__":
    unittest.main()
