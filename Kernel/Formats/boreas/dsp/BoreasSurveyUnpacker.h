//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2021 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/


#ifndef __dsp_BoreasSurveyUnpacker_h
#define __dsp_BoreasSurveyUnpacker_h

#include "ThreadContext.h"
#include "dsp/HistUnpacker.h"

namespace dsp {
  
  class BoreasSurveyUnpacker : public HistUnpacker
  {
  public:

    //! Constructor
    BoreasSurveyUnpacker (const char* name = "BoreasSurveyUnpacker");

    //! Destructor
    ~BoreasSurveyUnpacker ();

    //! Cloner (calls new)
    virtual BoreasSurveyUnpacker * clone () const;

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

#endif // __dsp_BoreasSurveyUnpacker_h
