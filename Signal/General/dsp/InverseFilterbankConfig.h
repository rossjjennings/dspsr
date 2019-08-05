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

#include "dsp/FilterbankConfig.h"
#include "dsp/InverseFilterbank.h"

namespace dsp
{
  class InverseFilterbank::Config : public Filterbank::Config
  {
  public:

    Config();

    InverseFilterbank* create();

    void set_input_overlap (unsigned n) { input_overlap = n; }

    unsigned get_input_overlap () const { return input_overlap; }

  protected:

    unsigned input_overlap;

  };

  //! Insertion operator
  std::ostream& operator << (std::ostream&, const InverseFilterbank::Config&);

  //! Extraction operator
  std::istream& operator >> (std::istream&, InverseFilterbank::Config&);
}

#endif
