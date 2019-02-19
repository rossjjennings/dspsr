//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2002 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Signal/General/dsp/FilterbankConfig.h

#ifndef __FilterbankConfig_h
#define __FilterbankConfig_h

#include "dsp/ConvolutionConfig.h"
#include "dsp/Filterbank.h"

namespace dsp
{
  class Filterbank::Config : public Convolution::Config
  {

  public:

    Config();

    //! Return a new Filterbank instance and configure it
    Filterbank* create ();

    //! Set the device on which the unpacker will operate
    void set_device (Memory*);

    //! Set the stream information for the device
    void set_stream (void*);

  };

}

#endif
