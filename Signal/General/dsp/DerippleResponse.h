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

    void calc_freq_response (unsigned n_freq);

    void set_fir_filter (const FIRFilter& _fir_filter) { fir_filter = _fir_filter; }

    const FIRFilter& get_fir_filter () const { return fir_filter; }

    const std::vector<float> get_freq_response () const { return freq_response; }

  protected:

    FIRFilter fir_filter;

    std::vector<float> freq_response;

    FTransform::Plan* forward;

  };
}

#endif
