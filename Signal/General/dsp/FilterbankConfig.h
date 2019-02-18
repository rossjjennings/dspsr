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
  {};

  //! Insertion operator
  std::ostream& operator << (std::ostream&, const Filterbank::Config&);

  //! Extraction operator
  std::istream& operator >> (std::istream&, Filterbank::Config&);
}

#endif
