//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2011 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __FilterbankEngine_h
#define __FilterbankEngine_h

#include "dsp/Filterbank.h"

namespace dsp {
  class Filterbank::Engine : public Reference::Able
  {
  public:

    Engine () { scratch = output = 0; }

    //! If kernel is not set, then the engine should set up for benchmark only
    virtual void setup (Filterbank*) = 0;

    //! provide some scratch space for the engine
    virtual void set_scratch (float *) = 0;

    //! Perform the filterbank operation on the input data
    virtual void perform (const dsp::TimeSeries * in,
                          dsp::TimeSeries * out,
                          uint64_t npart,
                          const uint64_t in_step,
                          const uint64_t out_step) = 0;

    //! Perform the filterbank operation on the input data
    virtual void perform (const dsp::TimeSeries * in,
                          dsp::TimeSeries* out,
                          dsp::TimeSeries* zero_DM_out,
                          uint64_t npart,
                          const uint64_t in_step,
                          const uint64_t out_step) = 0;


    //! Finish up
    virtual void finish () { }

    //! get the amount of scratch space the engine has calculated that it needs
    unsigned get_total_scratch_needed () const { return total_scratch_needed; }

    virtual void set_passband (dsp::Response* _passband) { passband = _passband; }

    virtual FTransform::Plan* get_forward () = 0;

    virtual FTransform::Plan* get_backward () = 0;

  protected:

    float* scratch;

    unsigned total_scratch_needed;

    float* output;
    unsigned output_span;

    dsp::Response* passband;

  };
}

#endif
