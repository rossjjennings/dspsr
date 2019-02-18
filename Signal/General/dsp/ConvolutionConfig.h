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
    //! Insertion operator
    friend std::ostream& operator << (std::ostream&, const Convolution::Config&);

    //! Extraction operator
    friend std::istream& operator >> (std::istream&, Convolution::Config&);

  public:

    //! When dedispersion takes place with respect to filterbank
    enum When
    {
      Before,
      During,
      After,
      Never
    };

    Config ();

    void set_nchan (unsigned n) { nchan = n; }
    unsigned get_nchan () const { return nchan; }

    void set_freq_res (unsigned n) { freq_res = n; }
    unsigned get_freq_res () const { return freq_res; }

    void set_convolve_when (When w) { when = w; }
    When get_convolve_when () const { return when; }

    //! Set the device on which the unpacker will operate
    void set_device (Memory*);

    //! Set the stream information for the device
    void set_stream (void*);

    //! Return a new Convolution instance and configure it
    Convolution* create ();

  protected:

    Memory* memory;
    void* stream;
    unsigned nchan;
    unsigned freq_res;
    When when;

  };


}

#endif
