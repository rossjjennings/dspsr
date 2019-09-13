//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2019 by Andrew Jameson and Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __ScalarFilter_h
#define __ScalarFilter_h

#include "dsp/Response.h"

namespace dsp {

  //! Simple rescaling with a scalar frequency response function
  class ScalarFilter: public Response {

  public:

    //! Default constructor
    ScalarFilter ();

    //! Destructor
    ~ScalarFilter ();

    //! Set the dimensions of the data, updating built attribute
    void resize(unsigned _npol, unsigned _nchan,
                unsigned _ndat, unsigned _ndim);

    //! Set the scalar factor to be applied in the response
    void set_scale_factor (float scale_factor);

    //! Set the number of input channels
    void set_nchan (unsigned _nchan);

    //! Set the length of the frequency response for each input channel
    void set_ndat (unsigned _ndat);

    //! Return the scale factor being applied
    float get_scale_factor ();
    float get_scale_factor () const;

    void build ();

    //! Create an Scalar filter for the specified observation
    void match (const Observation* input, unsigned nchan=0);

    //! Create an Scalar filter with the same number of channels as Response
    void match (const Response* response);

  protected:

    //! The scale factor to apply to the data
    float scale_factor;

  private:

    //! Set true when the filter has been built
    bool built;

  };

}

#endif
