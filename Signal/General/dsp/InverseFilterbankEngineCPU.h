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

    double setup_fft_plans (InverseFilterbank*);

    void set_scratch (float *);

    void perform (const dsp::TimeSeries* in, dsp::TimeSeries* out,
                  uint64_t npart, uint64_t in_step, uint64_t out_step);

    double get_scalefac() const {return scalefac;}

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

    bool verbose;

    Response* response;

    DerippleResponse* deripple;

  private:

    //! This is the number of floats per sample. This could be 1 or 2,
    //! depending on whether input is Analytic (complex) or Nyquist (real)
    unsigned n_per_sample;

    unsigned input_nchan;
    unsigned output_nchan;

    unsigned input_discard_neg;
    unsigned input_discard_pos;
    unsigned input_discard_total;

    unsigned output_discard_neg;
    unsigned output_discard_pos;
    unsigned output_discard_total;

    unsigned input_fft_length;
    unsigned output_fft_length;

    unsigned input_sample_step;
    unsigned output_sample_step;

    //! How much of the forward FFT to keep due to oversampling
    unsigned input_os_keep;
    //! How much of the forward FFT to discard due to oversampling
    unsigned input_os_discard;

    float* input_fft_scratch;
    float* output_fft_scratch;
    // float* response_stitch_scratch;
    // float* fft_shift_scratch;
    float* stitch_scratch;

    bool fft_plans_setup;

  };

}

#endif
