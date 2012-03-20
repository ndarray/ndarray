// -*- c++ -*-
/*
 * Copyright (c) 2010-2012, Jim Bosch
 * All rights reserved.
 *
 * ndarray is distributed under a simple BSD-like license;
 * see the LICENSE file that should be present in the root
 * of the source distribution, or alternately available at:
 * https://github.com/ndarray/ndarray
 */
%{
#include "ndarray.h"
#include "ndarray/swig.h"
#include "ndarray/swig/eigen.h"
#include <boost/scoped_ptr.hpp>
%}

#pragma SWIG nowarn=467

%define %declareNumPyConverters(TYPE...)
%typemap(out) TYPE {
    $result = ndarray::PyConverter< TYPE >::toPython($1);
}
%typemap(out) TYPE const &, TYPE &, TYPE const *, TYPE * {
    $result = ndarray::PyConverter< TYPE >::toPython(*$1);
}
%typemap(typecheck) TYPE, TYPE const *, TYPE const & {
    ndarray::PyPtr tmp($input,true);
    $1 = ndarray::PyConverter< TYPE >::fromPythonStage1(tmp);
    if (!($1)) PyErr_Clear();
}
%typemap(in) TYPE const & (TYPE val) {
    ndarray::PyPtr tmp($input,true);
    if (!ndarray::PyConverter< TYPE >::fromPythonStage1(tmp)) return NULL;
    if (!ndarray::PyConverter< TYPE >::fromPythonStage2(tmp, val)) return NULL;
    $1 = &val;
}
%typemap(in) TYPE {
    ndarray::PyPtr tmp($input,true);
    if (!ndarray::PyConverter< TYPE >::fromPythonStage1(tmp)) return NULL;
    if (!ndarray::PyConverter< TYPE >::fromPythonStage2(tmp, $1)) return NULL;
}
%enddef
