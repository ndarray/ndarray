import ndarray_mod
import unittest
import numpy

class TestNdArrayWrappers(unittest.TestCase):

    def testReturnArray(self):
        for suffix in ("11","10","22","21","20","33","32","31","30"):
            func = getattr(ndarray_mod, "returnArray_d%s" % suffix)
            cfunc = getattr(ndarray_mod, "returnArray_dc%s" % suffix)
            array = func()
            carray = cfunc()
            self.assert_((array == numpy.arange(0, array.size).reshape(array.shape)).all())
            self.assert_((carray == numpy.arange(0, carray.size).reshape(carray.shape)).all())
            self.assert_(array.flags["WRITEABLE"])
            self.assert_(not carray.flags["WRITEABLE"])

    def testAcceptArray(self):
        for suffix in ("11","10","22","21","20","33","32","31","30"):
            func = getattr(ndarray_mod, "acceptArray_d%s" % suffix)
            cfunc = getattr(ndarray_mod, "acceptArray_dc%s" % suffix)
            nd = int(suffix[0])
            shape = tuple(int(i) for i in numpy.random.randint(low=2,high=5,size=nd))
            array = numpy.zeros(shape, dtype=float)
            array[:] = numpy.arange(0, array.size).reshape(array.shape)
            self.assert_(func(array))
            self.assert_(cfunc(array))

    def testExtractArray(self):
        for suffix in ("11","10","22","21","20","33","32","31","30"):
            func = getattr(ndarray_mod, "extractArray_d%s" % suffix)
            cfunc = getattr(ndarray_mod, "extractArray_dc%s" % suffix)
            nd = int(suffix[0])
            shape = tuple(int(i) for i in numpy.random.randint(low=2,high=5,size=nd))
            array = numpy.zeros(shape, dtype=float)
            array[:] = numpy.arange(0, array.size).reshape(array.shape)
            self.assert_(func(array))
            self.assert_(cfunc(array))

    def testReturnVector(self):
        for suffix in ("2","3"):
            func = getattr(ndarray_mod, "returnVector_d%s" % suffix)
            vector = func()
            self.assertEqual(vector, tuple(numpy.arange(len(vector), dtype=float)))

    def testAcceptVector(self):
        for suffix in ("2","3"):
            func = getattr(ndarray_mod, "acceptVector_d%s" % suffix)
            nd = int(suffix[0])
            vector = tuple(numpy.arange(nd, dtype=float))
            self.assert_(func(vector))

    def testReturnEigenView(self):
        for suffix in ("11","10","22","21","20"):
            func = getattr(ndarray_mod, "returnEigenView_d%s" % suffix)
            cfunc = getattr(ndarray_mod, "returnEigenView_dc%s" % suffix)
            array = func()
            carray = cfunc()
            matrix = numpy.matrix(numpy.arange(0, array.size, dtype=float).reshape(array.shape))
            cmatrix = numpy.matrix(numpy.arange(0, carray.size, dtype=float).reshape(carray.shape))
            self.assert_((array == matrix).all())
            self.assert_((carray == cmatrix).all())
            self.assert_(array.flags["WRITEABLE"])
            self.assert_(not carray.flags["WRITEABLE"])

    def testReturnTransposedEigenView(self):
        for suffix in ("11","10","22","21","20"):
            func = getattr(ndarray_mod, "returnTransposedEigenView_d%s" % suffix)
            cfunc = getattr(ndarray_mod, "returnTransposedEigenView_dc%s" % suffix)
            array = func()
            carray = cfunc()
            if not suffix.startswith("1"):
                array = array.transpose()
                carray = carray.transpose()
            matrix = numpy.matrix(numpy.arange(0, array.size, dtype=float).reshape(array.shape))
            cmatrix = numpy.matrix(numpy.arange(0, carray.size, dtype=float).reshape(carray.shape))
            self.assert_((array == matrix).all())
            self.assert_((carray == cmatrix).all())
            self.assert_(array.flags["WRITEABLE"])
            self.assert_(not carray.flags["WRITEABLE"])

    def _testMemory(self):
        shape = (400, 400, 10)
        for n in range(1000000):
            a = ndarray_mod.makeArray_d33(shape)

if __name__=="__main__":
    unittest.main()
