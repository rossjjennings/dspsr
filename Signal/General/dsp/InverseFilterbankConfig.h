//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2002 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Signal/General/dsp/FilterbankConfig.h

#ifndef __InverseFilterbankConfig_h
#define __InverseFilterbankConfig_h

#include "dsp/InverseFilterbank.h"
#include "dsp/FilterbankConfig.h"

namespace dsp
{
  class InverseFilterbank::Config : Filterbank::Config
  {};

  //! Insertion operator
  std::ostream& operator << (std::ostream&, const Filterbank::Config&);

  //! Extraction operator
  std::istream& operator >> (std::istream&, Filterbank::Config&);
}

#endif
