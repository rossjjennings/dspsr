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

    //! Return true if the unpacker can operate on the specified device
    bool get_device_supported (Memory*) const;

    //! Set the device on which the unpacker will operate
    void set_device (Memory*);

    //! Engine used to perform unpacking on a device
    class Engine;
    void set_engine (Engine*);

    //! Return the pol for the digitizer index
    unsigned get_output_ipol (unsigned idig) const;

    //! Return the chan for the digitizer index
    unsigned get_output_ichan (unsigned idig) const;

    //! Return the number of dimensions per digitizer
    //unsigned get_ndim_per_digitizer () const;

    //! The quadrature components are offset by one
    unsigned get_output_offset(unsigned int) const;

  protected:
    
    //! Interface to alternate processing engine (e.g. GPU)
    Reference::To<Engine> engine;

    //! Bit table providing optimal scaling
    Reference::To<BitTable> table;

    //! Return true if we can convert the Observation
    bool matches (const Observation* observation);

    //! Unpack the BitSeries to a TimeSeries
    void unpack ();

  private:

    unsigned ndim;

    unsigned npol;

    bool device_prepared;

  };

  class LFAASPEADUnpacker::Engine : public Reference::Able
  {
  public:

    //! Unpack interface for engines from the input BitSeries to the output TimeSeries
    virtual void unpack(float scale, const BitSeries * input, TimeSeries * output) = 0;

    //! Called to perform engine configuration
    virtual void setup () = 0;
  };

}

#endif // __dsp_LFAASPEADUnpacker_h
