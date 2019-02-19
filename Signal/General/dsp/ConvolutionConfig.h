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
    friend std::ostream& operator << (std::ostream& os, const Convolution::Config& config)
    {
      os << config.get_nchan();
      if (config.get_convolve_when() == Convolution::Config::Before)
        os << ":B";
      else if (config.get_convolve_when() == Convolution::Config::During)
        os << ":D";
      else if (config.get_freq_res() != 1)
        os << ":" << config.get_freq_res();

      return os;
    }

    //! Extraction operator
    friend std::istream& operator >> (std::istream& is, Convolution::Config& config)
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
      else
      {
        unsigned nfft;
        is >> nfft;
        config.set_freq_res (nfft);
      }

      return is;
    }


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
    virtual void set_device (Memory* mem) { memory = mem ;} ;

    //! Set the stream information for the device
    virtual void set_stream (void* ptr) { stream = ptr; }

    //! Return a new Convolution instance and configure it
    virtual Convolution* create () = 0;

  protected:

    Memory* memory;
    void* stream;
    unsigned nchan;
    unsigned freq_res;
    When when;

  };


  // //! Insertion operator
  // std::ostream& operator << (std::ostream&, const Convolution::Config&);
  //
  // //! Extraction operator
  // std::istream& operator >> (std::istream&, Convolution::Config&);
}

#endif
