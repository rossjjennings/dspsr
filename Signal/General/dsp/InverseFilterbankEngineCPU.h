//-*-C++-*-

/***************************************************************************
 *
 *   Copyright (C) 2009 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Signal/General/dsp/InverseFilterbankEngineCPU.h

#ifndef __InverseFilterbankEngineCPU_h
#define __InverseFilterbankEngineCPU_h

#include "dsp/InverseFilterbankEngine.h"

#include "FTransform.h"

namespace dsp
{

  class InverseFilterbankEngineCPU : public dsp::InverseFilterbank::Engine
  {

  public:

    //! Default Constructor
    InverseFilterbankEngineCPU ();

    ~InverseFilterbankEngineCPU ();

    void setup (InverseFilterbank*);
    void set_scratch (float *);

    void perform (const dsp::TimeSeries* in, dsp::TimeSeries* out,
                  uint64_t npart, uint64_t in_step, uint64_t out_step);

    void finish ();

  protected:

    //! plan for computing forward fourier transforms
    FTransform::Plan* forward;

    //! plan for computing inverse fourier transforms
    FTransform::Plan* backward;

    //! Complex-valued data
    bool real_to_complex;

    //! device scratch sapce
    float* scratch;

    unsigned nchan_subband;
    unsigned freq_res;
    unsigned nfilt_pos;
    unsigned nkeep;

    bool verbose;

  };

}

#endif
