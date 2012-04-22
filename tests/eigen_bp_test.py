# -*- python -*-
#
#  Copyright (c) 2010-2012, Jim Bosch
#  All rights reserved.
#
#  ndarray is distributed under a simple BSD-like license;
#  see the LICENSE file that should be present in the root
#  of the source distribution, or alternately available at:
#  https://github.com/ndarray/ndarray
#
import eigen_bp_test_mod
import unittest
import numpy

class TestEigenWrappers(unittest.TestCase):

    def setUp(self):
        self.matrix_p_i = numpy.array([[1,2,3],[4,5,6]], dtype=numpy.int32, order="C")
        self.matrix_p_d = numpy.array([[1,2,3],[4,5,6]], dtype=numpy.float64, order="C")
        self.matrix_m_i = numpy.array([[1,2,3],[4,5,6]], dtype=numpy.int32, order="F")
        self.matrix_m_d = numpy.array([[1,2,3],[4,5,6]], dtype=numpy.float64, order="F")
        self.vector_i = numpy.array([1,2,3,4], dtype=numpy.int32)
        self.vector_d = numpy.array([1,2,3,4], dtype=numpy.float64)

    def testEigenView(self):
        for typechar in ("d", "i"):
            for name1 in ("M23", "MX3", "M2X", "MXX", "A23", "AX3", "A2X", "AXX"):
                for name2 in ("2p2", "2p1", "2p0"):
                    m = getattr(self, "matrix_p_%s" % typechar)
                    accept = getattr(eigen_bp_test_mod, "acceptEigenView_%s%s_%s" % (name1, typechar, name2))
                    ret = getattr(eigen_bp_test_mod, "returnEigenView_%s%s_%s" % (name1, typechar, name2))
                    self.assert_(accept(m))
                    self.assert_((ret() == m).all())
                for name2 in ("2m2", "2m1", "2p0"):
                    m = getattr(self, "matrix_m_%s" % typechar)
                    accept = getattr(eigen_bp_test_mod, "acceptEigenView_%s%s_%s" % (name1, typechar, name2))
                    ret = getattr(eigen_bp_test_mod, "returnEigenView_%s%s_%s" % (name1, typechar, name2))
                    self.assert_(accept(m))
                    self.assert_((ret() == m).all())

    def testAcceptMatrix(self):
        self.assert_(eigen_bp_test_mod.acceptMatrix_23d_cref(self.matrix_p_d))
        self.assert_(eigen_bp_test_mod.acceptMatrix_X3d_cref(self.matrix_p_d))
        self.assert_(eigen_bp_test_mod.acceptMatrix_2Xd_cref(self.matrix_p_d))
        self.assert_(eigen_bp_test_mod.acceptMatrix_XXd_cref(self.matrix_p_d))
        self.assert_(eigen_bp_test_mod.acceptMatrix_23d_cref(self.matrix_m_d))
        self.assert_(eigen_bp_test_mod.acceptMatrix_X3d_cref(self.matrix_m_d))
        self.assert_(eigen_bp_test_mod.acceptMatrix_2Xd_cref(self.matrix_m_d))
        self.assert_(eigen_bp_test_mod.acceptMatrix_XXd_cref(self.matrix_m_d))

    def testAcceptVector(self):
        self.assert_(eigen_bp_test_mod.acceptVector_41d_cref(self.vector_d))
        self.assert_(eigen_bp_test_mod.acceptVector_X1d_cref(self.vector_d))
        self.assert_(eigen_bp_test_mod.acceptVector_14d_cref(self.vector_d))
        self.assert_(eigen_bp_test_mod.acceptVector_1Xd_cref(self.vector_d))

    def testReturnMatrix(self):
        self.assert_((eigen_bp_test_mod.returnMatrix_23d() == self.matrix_p_d).all())
        self.assert_((eigen_bp_test_mod.returnMatrix_X3d() == self.matrix_p_d).all())
        self.assert_((eigen_bp_test_mod.returnMatrix_2Xd() == self.matrix_p_d).all())
        self.assert_((eigen_bp_test_mod.returnMatrix_XXd() == self.matrix_p_d).all())
        self.assert_((eigen_bp_test_mod.returnMatrix_23d_c() == self.matrix_p_d).all())
        self.assert_((eigen_bp_test_mod.returnMatrix_X3d_c() == self.matrix_p_d).all())
        self.assert_((eigen_bp_test_mod.returnMatrix_2Xd_c() == self.matrix_p_d).all())
        self.assert_((eigen_bp_test_mod.returnMatrix_XXd_c() == self.matrix_p_d).all())

    def testReturnObject(self):
        self.assert_((eigen_bp_test_mod.returnObject_23d() == self.matrix_p_d).all())
        self.assert_((eigen_bp_test_mod.returnObject_X3d() == self.matrix_p_d).all())
        self.assert_((eigen_bp_test_mod.returnObject_2Xd() == self.matrix_p_d).all())
        self.assert_((eigen_bp_test_mod.returnObject_XXd() == self.matrix_p_d).all())

    def testMatrixOwner(self):
        for suffix in ("23d", "X3d", "2Xd", "XXd"):
            cls = getattr(eigen_bp_test_mod, "MatrixOwner_%s" % suffix)
            obj = cls()
            self.assert_(obj.getMatrix_ref().base is obj)
            self.assert_(obj.getMatrix_cref().base is obj)
            self.assert_((obj.getMatrix_ref() == self.matrix_p_d).all())
            self.assert_((obj.getMatrix_cref() == self.matrix_p_d).all())
            self.assert_(obj.getMatrix_ref().flags["WRITEABLE"])
            self.assert_(not obj.getMatrix_cref().flags["WRITEABLE"])
            self.assert_(not obj.getMatrix_ref().flags["OWNDATA"])
            self.assert_(not obj.getMatrix_cref().flags["OWNDATA"])

if __name__=="__main__":
    unittest.main()
