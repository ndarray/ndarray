// -*- lsst-c++ -*-
/* 
 * LSST Data Management System
 * Copyright 2008, 2009, 2010 LSST Corporation.
 * 
 * This product includes software developed by the
 * LSST Project (http://www.lsst.org/).
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the LSST License Statement and 
 * the GNU General Public License along with this program.  If not, 
 * see <http://www.lsstcorp.org/LegalNotices/>.
 */

/**
 *  @file ndarray.i
 *  @brief Header file for ndarray SWIG/Python support.
 *
 *  \warning Both the Numpy C-API headers "arrayobject.h" and "ufuncobject.h" must
 *  be included before ndarray.i or any of the files in swig/ndarray/*
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
