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
import sys
sys.path.insert(0, os.path.abspath('Boost.NumPy'))
from SConsChecks import AddLibOptions, GetLibChecks

setupOptions, makeEnvironment, setupTargets, checks, libs = SConscript("Boost.NumPy/SConscript")
libs.extend(['swig', 'eigen', 'boost.test', 'fftw', 'boost.preprocessor'])
checks = GetLibChecks(libs)

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
    haveEigen = config.CheckEigen()
    haveFFTW = config.CheckFFTW()
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
    if haveSwig:
        pyEnv.AppendUnique(SWIGPATH = ["#include"])
    config.CheckBoostPP()
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
    print("Not building Boost.NumPy component.")

headers = SConscript(os.path.join("include", "SConscript"), exports="env")

prefix = Dir(GetOption("prefix")).abspath
install_headers = GetOption("install_headers")
if not install_headers:
    install_headers = os.path.join(prefix, "include")
for header in Flatten(headers):
    relative = os.path.relpath(header.abspath, Dir("#include").abspath)
    env.Alias("install", env.InstallAs(os.path.join(install_headers, relative), header))

tests = SConscript(os.path.join("tests", "SConscript"), exports=["env", "testEnv", "pyEnv", "bpEnv"])
