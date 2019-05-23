//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2019 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Signal/General/dsp/InverseFilterbankConfig.h

#ifndef __InverseFilterbankConfig_h
#define __InverseFilterbankConfig_h

#include <string>

#include "dsp/ConvolutionConfig.h"
#include "dsp/InverseFilterbank.h"
// #include "dsp/Response.h"
// #include "dsp/TimeSeries.h"
#include "dsp/Apodization.h"

namespace dsp
{
  class InverseFilterbank::Config : public Convolution::Config
  {
  public:

    Config();

    //! Return a new InverseFilterbank instance and configure it
    InverseFilterbank* create ();

    //! Set the device on which the unpacker will operate
    // void set_device (Memory*);

    //! Set the stream information for the device
    // void set_stream (void*);
    //! Extraction operator
    friend std::istream& operator >> (std::istream& is, InverseFilterbank::Config& config)
    {
      unsigned value;
      is >> value;

      config.set_nchan (value);
      config.set_convolve_when (Convolution::Config::After);

      if (is.eof())
        return is;

      if (is.peek() != ':')
      {
        is.fail();
        return is;
      }

      // throw away the colon
      is.get();

      if (is.peek() == 'D' || is.peek() == 'd')
      {
        is.get();  // throw away the D
        config.set_convolve_when (Convolution::Config::During);
      }
      else if (is.peek() == 'B' || is.peek() == 'b')
      {
        is.get();  // throw away the B
        config.set_convolve_when (Convolution::Config::Before);
      }
      else if (is.peek() == 'A' || is.peek() == 'a')
      {
        // even though this is the default, make it possible to pass A anyways.
        is.get();  // throw away the A
        config.set_convolve_when (Convolution::Config::After);
      }
      else
      {
        unsigned nfft;
        is >> nfft;
        config.set_freq_res (nfft);
      }

      if (is.peek() == ':') {
        is.get();
        unsigned overlap;
        is >> overlap;
        config.set_input_overlap (overlap);
      }

      return is;
    }


    void set_input_overlap (unsigned n) { input_overlap = n; }

    unsigned get_input_overlap () const { return input_overlap; }

  protected:

    unsigned input_overlap;


  };
}

#endif
