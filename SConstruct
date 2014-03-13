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

import os

def CheckBoostTest(context):
    source_file = """
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE config
#include "boost/test/unit_test.hpp"
#include "boost/test/results_reporter.hpp"
#include <iostream>

BOOST_AUTO_TEST_CASE(ConfigTestCase) {
    boost::unit_test::results_reporter::set_stream(std::cout);
    BOOST_CHECK(true);
}
"""
    context.Message("Checking for Boost.Test Library...")
    context.env.setupPaths(
        prefix = GetOption("boost_prefix"),
        include = GetOption("boost_include"),
        lib = GetOption("boost_lib")
        )
    result = (
        context.checkLibs([''], source_file) or
        context.checkLibs(['boost_unit_test_framework'], source_file) or
        context.checkLibs(['boost_unit_test_framework-mt'], source_file)
        )
    if not result:
        context.Result(0)
        print "Cannot build against Boost.Test."
        return False
    result, output = context.TryRun(source_file, '.cpp')
    if not result:
        context.Result(0)
        print "Cannot build against Boost.Test."
        return False
    context.Result(1)
    return True

def CheckSwig(context):
    context.Message("Checking for SWIG...")
    context.env.PrependUnique(SWIGFLAGS = ["-python", "-c++"])
    result, swig_cmd = config.TryAction("which swig > $TARGET")
    context.Result(result)
    if result:
        context.env.AppendUnique(SWIGPATH = ["#include"])
        print "Using SWIG at", swig_cmd.strip()
    return result

setupOptions, makeEnvironment, setupTargets, checks = SConscript("Boost.NumPy/SConscript")

checks["CheckBoostTest"] = CheckBoostTest
checks["CheckSwig"] = CheckSwig

variables = setupOptions()

AddOption("--with-eigen", dest="eigen_prefix", type="string", nargs=1, action="store",
          metavar="DIR", default=os.environ.get("EIGEN_DIR"),
          help="prefix for Eigen libraries; should have an 'include' subdirectory")
AddOption("--with-eigen-include", dest="eigen_include", type="string", nargs=1, action="store",
          metavar="DIR", help="location of Eigen header files")

AddOption("--with-fftw", dest="fftw_prefix", type="string", nargs=1, action="store",
          metavar="DIR", default=os.environ.get("FFTW_DIR"),
          help="prefix for FFTW libraries; should have 'include' and 'lib' subdirectories")
AddOption("--with-fftw-include", dest="fftw_include", type="string", nargs=1, action="store",
          metavar="DIR", help="location of FFTW header files")
AddOption("--with-fftw-lib", dest="fftw_lib", type="string", nargs=1, action="store",
          metavar="DIR", help="location of FFTW library")

building = not GetOption("help") and not GetOption("clean")

env = makeEnvironment(variables)
env.AppendUnique(CPPPATH=["#include"])

if building:
    config = env.Configure(custom_tests=checks)
    config.env.setupPaths(
        prefix = GetOption("eigen_prefix"),
        include = GetOption("eigen_include"),
        lib = None
        )
    config.env.setupPaths(
        prefix = GetOption("fftw_prefix"),
        include = GetOption("fftw_include"),
        lib = GetOption("fftw_lib")
        )
    haveEigen = config.CheckCXXHeader("Eigen/Core")
    haveFFTW = config.CheckLibWithHeader("fftw3", "fftw3.h", "C", autoadd=False)
    env = config.Finish()
else:
    haveEigen = False
    haveFFTW = False
env.haveEigen = haveEigen
env.haveFFTW = haveFFTW

testEnv = env.Clone()
if building:
    config = testEnv.Configure(custom_tests=checks)
    haveBoostTest = config.CheckBoostTest()
    testEnv = config.Finish()
else:
    haveBoostTest = False
testEnv.haveBoostTest = haveBoostTest

pyEnv = env.Clone()
if building:
    config = pyEnv.Configure(custom_tests=checks)
    havePython = config.CheckPython() and config.CheckNumPy()
    haveSwig = havePython and config.CheckSwig()
    pyEnv = config.Finish()
else:
    havePython = False
    haveSwig = False
pyEnv.havePython = havePython
pyEnv.haveSwig = haveSwig

bpEnv = pyEnv.Clone()
if building:
    config = bpEnv.Configure(custom_tests=checks)
    haveBoostPython = config.CheckBoostPython()
    bpEnv = config.Finish()
else:
    haveBoostPython = False
bpEnv.haveBoostPython = haveBoostPython

bpEnv.AppendUnique(CPPPATH=["#Boost.NumPy"])
if haveBoostPython:
    setupTargets(bpEnv, root="Boost.NumPy")
elif building:
    print "Not building Boost.NumPy component."

headers = SConscript(os.path.join("include", "SConscript"), exports="env")
prefix = Dir(GetOption("prefix")).abspath
install_headers = GetOption("install_headers")
if not install_headers:
    install_headers = os.path.join(prefix, "include")
for header in Flatten(headers):
    relative = os.path.relpath(header.abspath, Dir("#include").abspath)
    env.Alias("install", env.InstallAs(os.path.join(install_headers, relative), header))

# test builds temporarily disabled; something weird is going on with SCons
if False:
    tests = SConscript(os.path.join("tests", "SConscript"), exports=["env", "testEnv", "pyEnv", "bpEnv"])
