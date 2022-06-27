//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2022 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/


#ifndef __dsp_BoreasVoltageUnpacker_h
#define __dsp_BoreasVoltageUnpacker_h

#include "ThreadContext.h"
#include "dsp/HistUnpacker.h"

namespace dsp {
  
  class BoreasVoltageUnpacker : public HistUnpacker
  {
  public:

    //! Constructor
    BoreasVoltageUnpacker (const char* name = "BoreasVoltageUnpacker");

    //! Destructor
    ~BoreasVoltageUnpacker ();

    //! Cloner (calls new)
    virtual BoreasVoltageUnpacker * clone () const;

    //! Return the pol for the digitizer index
    unsigned get_output_ipol (unsigned idig) const;

    //! Return the chan for the digitizer index
    unsigned get_output_ichan (unsigned idig) const;

    //! Return the number of dimensions per digitizer
    //unsigned get_ndim_per_digitizer () const;

    //! The quadrature components are offset by one
    unsigned get_output_offset(unsigned int) const;

  protected:
    
    //! Return true if we can convert the Observation
    bool matches (const Observation* observation);

    //! Unpack the BitSeries to a TimeSeries
    void unpack ();

  private:

    unsigned ndim;

    unsigned npol;

    bool device_prepared;

  };

}

#endif // __dsp_BoreasVoltageUnpacker_h
