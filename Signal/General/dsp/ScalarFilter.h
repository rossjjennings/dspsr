//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2018 by Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Signal/General/dsp/ScalarFilter.h

#ifndef __ScalarFilter_h
#define __ScalarFilter_h

#include "dsp/Response.h"

namespace dsp {

  //! Simple data rescaling a scalar frequency response function
  class ScalarFilter: public Response {

  public:

    //! Default constructor
    ScalarFilter ();

    //! Destructor
    ~ScalarFilter ();

    //! Create an Scalar filter for the specified observation
    void match (const Observation* input, unsigned nchan);

    //! Create an Scalar filter with the same number of channels as Response
    void match (const Response* response);

    void set_scale (float scale_factor);

    void operate (float* data, unsigned ipol, int ichan) const;

    //! Multiply spectrum by scalar frequency response
    void operate (float* spectrum, unsigned poln, int ichan_start, unsigned nchan_op) const;


  protected:

    //! The scale factor to apply to the data
    float scale_factor;

  private:

    //! Set true when the response has been calculated
    bool calculated;

  };
  
}

#endif
