//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2019 by Dean Shaff and Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
 // dspsr/Signal/General/dsp/DerippleResponse.h

#ifndef __DerippleResponse_h
#define __DerippleResponse_h

#include <vector>

#include "dsp/Response.h"
#include "dsp/FIRFilter.h"

namespace dsp {
  class DerippleResponse : public Response {

  public:

    DerippleResponse ();

    void prepare (const Observation* input, unsigned channels);

    void match (const Observation* input, unsigned channels);

    void set_optimal_ndat ();

    void calc_freq_response (std::vector<float>& freq_response, unsigned n_freq);

    void set_fir_filter (const FIRFilter& _fir_filter) { fir_filter = _fir_filter; }

    const FIRFilter& get_fir_filter () const { return fir_filter; }

    //! Set the number of channels into which the band will be divided
    virtual void set_nchan (unsigned nchan);

    //! Set the frequency resolution this many times the minimum required
    void set_times_minimum_nfft (unsigned times);

    //! Set the frequency resolution in each channel of the kernel
    void set_frequency_resolution (unsigned nfft);

    //! Get the frequency resolution in each channel of the kernel
    unsigned get_frequency_resolution () const { return ndat; }


  protected:

    void build ();

    //! FIR filter that contains time domain filter coefficients
    FIRFilter fir_filter;

    //! FFT plan for computing frequency response of FIR filter
    FTransform::Plan* forward;

    //! flag indicating whether frequency response has been built
    bool built;

    //! Flag set when set_frequency_resolution() method is called
    bool frequency_resolution_set;

    //! Choose filter length this many times the minimum length
    unsigned times_minimum_nfft;
  };
}

#endif
