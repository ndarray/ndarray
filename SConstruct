import os
import sys
import lsst.SConsUtils as scons
import eups

dependencies = ["boost", "python", "eigen", "fftw"]  # utils also needed, but only for scons

env = scons.makeEnv("ndarray",
                    r"$HeadURL$",
                    [["boost", "boost/shared_ptr.hpp"],
                     ["boost", "boost/test/unit_test.hpp", "boost_unit_test_framework:C++"],
                     ["python", "Python.h"],
                     ["eigen", "Eigen/Core.h"],
                     ["fftw", "fftw3.h", "fftw3"]
                     ])

import numpy
env.Append(CPPPATH=[numpy.get_include()])

env.Help("""
Multidimensional array and NumPy support package for C++
""")

###############################################################################
# Boilerplate below here

pkg = env["eups_product"]
env.libs[pkg] = env.getlibs(" ".join(dependencies))
m4flags = "-I%s" % os.path.join(os.path.abspath(eups.productDir(pkg)), 'm4')

#
# Build/install things
#
generated = ["#include/lsst/ndarray/ArrayRef.h",
             "#include/lsst/ndarray/operators.h",
             "#include/lsst/ndarray/Vector.h",
             "#include/lsst/ndarray/fft/FFTWTraits.h",
             ]
headers = [env.M4(filename, "%s.m4" % filename, M4FLAGS=m4flags) for filename in generated]

for d in (
    ".",
    "doc",
    "tests",
):
    if d != ".":
        try:
            SConscript(os.path.join(d, "SConscript"))
        except Exception, e:
            print >> sys.stderr, "In processing file %s:" % (os.path.join(d, "SConscript"))
            print >> sys.stderr, e
    Clean(d, Glob(os.path.join(d, "*~")))
    Clean(d, Glob(os.path.join(d, "*.pyc")))

env['IgnoreFiles'] = r"(~$|\.pyc$|^\.svn$|\.o$)"

Alias("install", [
    env.Install(env['prefix'], "doc"),
    env.Install(env['prefix'], "include"),
    env.Install(env['prefix'], "tests"),
    env.Install(env['prefix'], "python"),
    env.InstallEups(os.path.join(env['prefix'], "ups")),
])

scons.CleanTree(r"*~ core *.so *.os *.o")

#
# Build TAGS files
#
files = scons.filesToTag()
if files:
    env.Command("TAGS", files, "etags -o $TARGET $SOURCES")

env.Declare()
