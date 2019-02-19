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

#include "dsp/ConvolutionConfig.h"
#include "dsp/InverseFilterbank.h"

namespace dsp
{
  class InverseFilterbank::Config : public Convolution::Config
  {
  public:

    Config();

    //! Return a new InverseFilterbank instance and configure it
    InverseFilterbank* create ();

    //! Set the device on which the unpacker will operate
    void set_device (Memory*);

    //! Set the stream information for the device
    void set_stream (void*);


  };
  //! Insertion operator
  std::ostream& operator << (std::ostream&, const InverseFilterbank::Config&);

  //! Extraction operator
  std::istream& operator >> (std::istream&, InverseFilterbank::Config&);

}

#endif
