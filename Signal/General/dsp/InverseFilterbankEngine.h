//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2011 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __FilterbankEngine_h
#define __FilterbankEngine_h

#include "dsp/InverseFilterbank.h"
//#include "dsp/filterbank_engine.h"

//! Abstract base class for derived engines that operate on data
//! in order to perform  inverse (synthesis) filterbank operation 
class dsp::InverseFilterbank::Engine : public Reference::Able
{
public:

  Engine () { scratch = output = 0; }

  //! If kernel is not set, then the engine should set up for benchmark only
  virtual void setup (InverseFilterbank*) = 0;

  //! provide some scratch space for the engine
  virtual void set_scratch (float *) = 0;

  //! Perform the filterbank operation on the input data
  virtual void perform (const dsp::TimeSeries * in,
                        dsp::TimeSeries * out,
                        uint64_t npart,
                        const uint64_t in_step,
                        const uint64_t out_step) = 0;

  //! Finish up
  virtual void finish () { }

protected:

  float* scratch;

  float* output;
  unsigned output_span;

};

#endif
