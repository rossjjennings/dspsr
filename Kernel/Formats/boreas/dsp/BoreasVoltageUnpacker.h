/*

 */

#ifndef __dsp_BoreasVoltageUnpacker_h
#define __dsp_BoreasVoltageUnpacker_h

#include "dsp/Unpacker.h"
#include "ThreadContext.h"

namespace dsp {
  
  class BoreasVoltageUnpacker : public Unpacker
  {
  public:

    //! Constructor
    BoreasVoltageUnpacker (const char* name = "BoreasVoltageUnpacker");

    //! Destructor
    ~BoreasVoltageUnpacker ();

    //! Cloner (calls new)
    virtual BoreasVoltageUnpacker * clone () const;

    //! Return true if the unpacket supports the specified output ordering
    bool get_order_supported (TimeSeries::Order order) const;

    //! Set the ordering of the output time series
    void set_output_order (TimeSeries::Order order);

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

    //! Return true if we can convert the Observation
    bool matches (const Observation* observation);

    void unpack ();

  private:

    unsigned ndim;

    unsigned npol;

    bool device_prepared;

  };

  class BoreasVoltageUnpacker::Engine : public Reference::Able
  {
    public:

      virtual bool get_device_supported (Memory* memory) const = 0;

      virtual void setup () = 0;

      virtual void unpack(const BitSeries * input, TimeSeries * output) = 0;

  };

}

#endif
