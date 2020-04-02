//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2011 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __InverseFilterbankEngine_h
#define __InverseFilterbankEngine_h

#include <functional>

#include "dsp/InverseFilterbank.h"
#include "EventEmitter.h"
#include "Functor.h"

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

  virtual void perform (const dsp::TimeSeries * in,
                        dsp::TimeSeries * out,
                        dsp::TimeSeries* zero_DM_out,
                        uint64_t npart,
                        const uint64_t in_step,
                        const uint64_t out_step) = 0;

  //! Finish up
  virtual void finish () { }

  //! get the amount of scratch space the engine has calculated that it needs
  unsigned get_total_scratch_needed () const { return total_scratch_needed; }

  bool get_report () const { return report; }

  void set_report (bool _report) { report = _report; }

  class Reporter {
  public:
    virtual void operator() (float*, unsigned, unsigned, unsigned, unsigned) {};
  };

  // A event emitter that takes a data array, and the nchan, npol, ndat and ndim
  // associated with the data array
  EventEmitter<Reporter> reporter;


protected:

  float* scratch;

  float* output;
  unsigned output_span;

  unsigned total_scratch_needed;

  //! Flag indicating whether to report intermediate data products
  bool report;


};

#endif
