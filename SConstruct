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

setupOptions, makeEnvironment, setupTargets, checks = SConscript("Boost.NumPy/SConscript")

setupOptions()

AddOption("--with-eigen", dest="eigen_prefix", type="string", nargs=1, action="store",
          metavar="DIR", default=os.environ.get("EIGEN_DIR"),
          help="prefix for Eigen libraries; should have an 'include' subdirectory")
AddOption("--with-eigen-include", dest="eigen_include", type="string", nargs=1, action="store",
          metavar="DIR", help="location of Eigen header files")

building = not GetOption("help") and not GetOption("clean")

env = makeEnvironment()
env.AppendUnique(CPPPATH=["#include"])

if building:
   config = env.Configure()
   eigen_prefix = GetOption("eigen_prefix")
   eigen_include = GetOption("eigen_include")
   if eigen_prefix is not None:
      if eigen_include is None:
         eigen_include = os.path.join(eigen_prefix, "include")
   if eigen_include:
      context.env.AppendUnique(CPPPATH=[eigen_include])
   haveEigen = config.CheckCXXHeader("Eigen/Core")
   env = config.Finish()
else:
   haveEigen = False
env.haveEigen = haveEigen

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
else:
   print "Not building Boost.NumPy component."

headers = SConscript(os.path.join("include", "SConscript"), exports="env")
prefix = Dir(GetOption("prefix")).abspath
for header in Flatten(headers):
   relative = os.path.relpath(header.abspath, Dir("#include").abspath)
   env.Alias("install", env.Install(os.path.join(prefix, "include", relative, relative), header))

#tests = SConscript(os.path.join("tests", "SConscript"), exports=["env", "pyEnv", "bpEnv"])
