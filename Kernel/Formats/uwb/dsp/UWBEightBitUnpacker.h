//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2019 by Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __dsp_UWBEightBitUnpacker_h
#define __dsp_UWBEightBitUnpacker_h

#include "dsp/HistUnpacker.h"

namespace dsp {
  
  class UWBEightBitUnpacker : public HistUnpacker
  {
  public:

    //! Constructor
    UWBEightBitUnpacker (const char* name = "UWBEightBitUnpacker");

    //! Destructor
    ~UWBEightBitUnpacker ();

    bool get_order_supported (TimeSeries::Order order) const;

    void set_output_order (TimeSeries::Order order);

    unsigned get_output_offset (unsigned idig) const;

    unsigned get_output_ipol (unsigned idig) const;

    unsigned get_output_ichan (unsigned idig) const;
    
    unsigned get_ndim_per_digitizer () const;

    //! Cloner (calls new)
    virtual UWBEightBitUnpacker * clone () const;

  protected:
    
    //! Return true if we can convert the Observation
    bool matches (const Observation* observation);

    void get_scales_and_offsets ();

    void unpack ();

  private:

    unsigned ndim;

    unsigned npol;

    bool have_scales_and_offsets;

    std::vector<std::vector<float> > offsets;

    std::vector<std::vector<float> > scales;

  };

}

#endif
