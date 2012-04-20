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
    boost_prefix = GetOption("boost_prefix")
    boost_include = GetOption("boost_include")
    boost_lib = GetOption("boost_lib")
    if boost_prefix is not None:
        if boost_include is None:
            boost_include = os.path.join(boost_prefix, "include")
        if boost_lib is None:
            boost_lib = os.path.join(boost_prefix, "lib")
    if boost_include:
        context.env.PrependUnique(CPPPATH=[boost_include])
    if boost_lib:
        context.env.PrependUnique(LIBPATH=[boost_lib])
    result = (
        CheckLibs(context, [''], source_file) or
        CheckLibs(context, ['boost_unit_test_framework'], source_file) or
        CheckLibs(context, ['boost_unit_test_framework-mt'], source_file)
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

setupOptions, makeEnvironment, setupTargets, checks, CheckLibs = SConscript("Boost.NumPy/SConscript")

checks["CheckBoostTest"] = CheckBoostTest

variables = setupOptions()

AddOption("--with-eigen", dest="eigen_prefix", type="string", nargs=1, action="store",
          metavar="DIR", default=os.environ.get("EIGEN_DIR"),
          help="prefix for Eigen libraries; should have an 'include' subdirectory")
AddOption("--with-eigen-include", dest="eigen_include", type="string", nargs=1, action="store",
          metavar="DIR", help="location of Eigen header files")

building = not GetOption("help") and not GetOption("clean")

env = makeEnvironment(variables)
env.AppendUnique(CPPPATH=["#include"])

if building:
    config = env.Configure()
    eigen_prefix = GetOption("eigen_prefix")
    eigen_include = GetOption("eigen_include")
    if eigen_prefix is not None:
        if eigen_include is None:
            eigen_include = os.path.join(eigen_prefix, "include")
    if eigen_include:
        config.env.AppendUnique(CPPPATH=[eigen_include])
    haveEigen = config.CheckCXXHeader("Eigen/Core")
    env = config.Finish()
else:
    haveEigen = False
env.haveEigen = haveEigen

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
    pyEnv = config.Finish()
else:
    havePython = False
pyEnv.havePython = havePython

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

tests = SConscript(os.path.join("tests", "SConscript"), exports=["env", "testEnv", "pyEnv", "bpEnv"])
