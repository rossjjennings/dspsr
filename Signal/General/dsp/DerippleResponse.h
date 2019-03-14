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

    //! Default constructor
    DerippleResponse ();

    //! Destructor
    ~DerippleResponse ();

    //! Set the dimensions of the data, updating built attribute
    void resize(unsigned _npol, unsigned _nchan,
                unsigned _ndat, unsigned _ndim);

    void match (const Observation* input, unsigned channels, const Rational& osf);

    void match (const Observation* input, unsigned channels);

    //! Create a DerippleResponse with the same number of channels as Response
    void match (const Response* response);

    //! Set the number of input channels
    void set_nchan (unsigned _nchan);

    //! Set the length of the frequency response for each input channel
    void set_ndat (unsigned _ndat);

    void build ();

    //! Calculate the frequency response, filling up freq_response vector.
    void calc_freq_response (std::vector<float>& freq_response, unsigned n_freq);

    void set_fir_filter (const FIRFilter& _fir_filter) { fir_filter = _fir_filter; }

    const FIRFilter& get_fir_filter () const { return fir_filter; }

  protected:


    //! FIR filter that contains time domain filter coefficients
    FIRFilter fir_filter;

    //! FFT plan for computing frequency response of FIR filter
    FTransform::Plan* forward;

    //! flag indicating whether frequency response has been built
    bool built;
  };
}

#endif
