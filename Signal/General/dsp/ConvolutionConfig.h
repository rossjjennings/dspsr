//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2019 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Signal/General/dsp/ConvolutionConfig.h

#ifndef __ConvolutionConfig_h
#define __ConvolutionConfig_h

#include "dsp/Convolution.h"

namespace dsp
{
  class Convolution::Config
  {


  public:

    //! When dedispersion takes place
    enum When
    {
      Before,
      During,
      After,
      Never
    };

    Config () {};

    virtual void set_nchan (unsigned n) { nchan = n; }
    virtual unsigned get_nchan () const { return nchan; }

    virtual void set_freq_res (unsigned n) { freq_res = n; }
    virtual unsigned get_freq_res () const { return freq_res; }

    virtual void set_convolve_when (When w) { when = w; }
    virtual When get_convolve_when () const { return when; }

    //! Set the device on which the unpacker will operate
    virtual void set_device (Memory*) = 0;

    //! Set the stream information for the device
    virtual void set_stream (void*) = 0;

    //! Return a new Convolution instance and configure it
    virtual Convolution* create () = 0;

  protected:

    Memory* memory;
    void* stream;
    unsigned nchan;
    unsigned freq_res;
    When when;

  };

  //! Insertion operator
  std::ostream& operator << (std::ostream&, const Convolution::Config&);

  //! Extraction operator
  std::istream& operator >> (std::istream&, Convolution::Config&);
}

#endif
