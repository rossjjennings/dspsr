//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2020 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/


#ifndef __dsp_LFAASPEADUnpacker_h
#define __dsp_LFAASPEADUnpacker_h

#include "dsp/EightBitUnpacker.h"
#include "ThreadContext.h"

namespace dsp {
  
  class LFAASPEADUnpacker : public HistUnpacker
  {
  public:

    //! Constructor
    LFAASPEADUnpacker (const char* name = "LFAASPEADUnpacker");

    //! Destructor
    ~LFAASPEADUnpacker ();

    //! Cloner (calls new)
    virtual LFAASPEADUnpacker * clone () const;

  protected:
    
    Reference::To<BitTable> table;

    //! Return true if we can convert the Observation
    bool matches (const Observation* observation);

    unsigned get_output_offset (unsigned idig) const;
    unsigned get_output_ipol (unsigned idig) const;
    unsigned get_output_ichan (unsigned idig) const;

    void unpack ();

  private:

    unsigned ndim;

    unsigned npol;

  };
}

#endif
