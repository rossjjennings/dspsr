//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2004 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Kernel/Classes/dsp/FloatUnpacker.h

#ifndef __FloatUnpacker_h
#define __FloatUnpacker_h

#include "dsp/Unpacker.h"

namespace dsp {

  //! Unpacks floating point numbers stored in time-major order
  class FloatUnpacker: public Unpacker
  {

  public:

    //! Null constructor
    FloatUnpacker (const char* name = "FloatUnpacker");

    //! The unpacking routine
    void unpack ();

    //! Return true if we can convert the Observation
    bool matches (const Observation* observation);

    //! Return true if the output order is supported
    bool get_order_supported (TimeSeries::Order) const;

    //! Set the order of the dimensions in the output TimeSeries
    void set_output_order (TimeSeries::Order);

    //! Return true if the unpacker can operate on the specified device
    bool get_device_supported (Memory*) const;

    //! Set the device on which the unpacker will operate
    void set_device (Memory*);

    //! Engine used to perform unpacking
    class Engine;

    //! Set the Engine to be used
    void set_engine (Engine*);

  protected:

    //! Interface to alternate processing engine (e.g. GPU)
    Reference::To<Engine> engine;

  private:

    //! Flag if 
    bool device_prepared;

  };

  class FloatUnpacker::Engine : public Reference::Able
  {
    public:

      virtual bool get_device_supported (Memory* memory) const = 0;

      virtual void setup () = 0;

      virtual void unpack(const BitSeries * input, TimeSeries * output) = 0;

  };

}

#endif
