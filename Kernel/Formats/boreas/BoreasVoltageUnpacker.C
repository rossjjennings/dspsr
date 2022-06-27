//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2022 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "dsp/BoreasVoltageUnpacker.h"

#if HAVE_CUDA
#include "dsp/MemoryCUDA.h"
#include "dsp/BoreasVoltageUnpackerCUDA.h"
#include <cuda_runtime.h>
#endif

#include "Error.h"

using namespace std;

dsp::BoreasVoltageUnpacker::BoreasVoltageUnpacker (const char* _name) : Unpacker (_name)
{
  if (verbose)
    cerr << "dsp::BoreasVoltageUnpacker ctor" << endl;
}

dsp::BoreasVoltageUnpacker::~BoreasVoltageUnpacker ()
{
}

dsp::BoreasVoltageUnpacker * dsp::BoreasVoltageUnpacker::clone () const
{
  return new BoreasVoltageUnpacker (*this);
}

void dsp::BoreasVoltageUnpacker::set_engine (Engine* _engine)
{
  if (verbose)
    cerr << "dsp::BoreasVoltageUnpacker::set_engine" << endl;
  engine = _engine;
}

//! Return true if the unpacker can operate on the specified device
bool dsp::BoreasVoltageUnpacker::get_device_supported (Memory* memory) const
{
  bool supported = false;
  if (engine)
    supported = engine->get_device_supported (memory);
  if (verbose)
    cerr << "dsp::BoreasVoltageUnpacker::get_device_supported supported=" << supported << endl;
  return supported;
}

//! Set the device on which the unpacker will operate
void dsp::BoreasVoltageUnpacker::set_device (Memory* memory)
{
  if (verbose)
    cerr << "dsp::BoreasVoltageUnpacker::set_device()" << endl;
#if HAVE_CUDA
  CUDA::DeviceMemory * gpu_mem = dynamic_cast< CUDA::DeviceMemory*>( memory );
  if (gpu_mem)
  {
    cudaStream_t stream = gpu_mem->get_stream();
    set_engine (new CUDA::BoreasVoltageUnpackerEngine(stream));
  }
#endif

  if (engine)
    engine->setup ();
  else
    Unpacker::set_device (memory);

  device_prepared = true;
}

bool dsp::BoreasVoltageUnpacker::matches (const Observation* observation)
{
  return observation->get_machine() == "Boreas"
    && observation->get_ndim() == 2
    && observation->get_npol() == 2
    && (observation->get_nbit() == 16 || observation->get_nbit() == 32);
}

bool dsp::BoreasVoltageUnpacker::get_order_supported (TimeSeries::Order order) const
{
  return order == TimeSeries::OrderFPT || order == TimeSeries::OrderFPT;
}

void dsp::BoreasVoltageUnpacker::set_output_order (TimeSeries::Order order)
{
  output_order = order;
}

void dsp::BoreasVoltageUnpacker::unpack ()
{
  if (output->get_order() != TimeSeries::OrderFPT)
    throw Error (InvalidState, "dsp::BoreasVoltageUnpacker::unpack", "cannot unpack into FPT order");

  if (engine)
  {
    if (verbose)
      cerr << "dsp::BoreasVoltageUnpacker::unpack using Engine" << endl;
    engine->unpack(input, output);
    return;
  }
  if (verbose)
    cerr << "dsp::BoreasVoltageUnpacker::unpack using CPU" << endl;

  float * from32 = (float*) input->get_rawptr();
  int16_t * from16 = (int16_t *) input->get_rawptr();
  float * into;

  const uint64_t ndat = input->get_ndat();
  const unsigned nchan = input->get_nchan();
  const unsigned npol  = input->get_npol();
  const unsigned ndim  = 2;
  const unsigned nbit = input->get_nbit();
  const unsigned ndat_per_heap = 384;
  const unsigned nheap = unsigned(ndat / ndat_per_heap);

  // data are stored in Heaps of 384 samples, each of which are packed in SFPT order

  // some programs (digifil) do not call set_device
  if (! device_prepared )
    set_device ( Memory::get_manager ());

  if (verbose)
    cerr << "dsp::BoreasVoltageUnpacker::unpack nheap=" << nheap << " ndat=" << ndat << " nchan=" << nchan << " npol=" << npol << endl;

  switch ( output->get_order() )
  {
    case TimeSeries::OrderFPT:
    {
#ifdef _DEBUG 
      cerr << "dsp::BoreasVoltageUnpacker::unpack TimeSeries::OrderFPT" << endl;
#endif
      //float * into_base = output->get_datptr (0, 0);
      for (unsigned iheap=0; iheap<nheap; iheap++)
      {
        for (unsigned ichan=0; ichan<nchan; ichan++)
        {
          for (unsigned ipol=0; ipol<npol; ipol++)
          {
            into = output->get_datptr (ichan, ipol) + iheap * (ndat_per_heap * ndim);
            //cerr << "[" << ichan << "][" << ipol << "][" << iheap << "] offset=" << (into - into_base) << endl;
            for (unsigned idat=0; idat<ndat_per_heap; idat++)
            {
              //if (iheap == 0 && ichan == 0)
              //  cerr << "ipol=" << ipol << " input=" << from32[0] << "," << from32[1] << endl;
              //if (from[0] > 32000 || from[0] < -32000 || from[1] > 32000 || from[1] < -32000)
              //  cerr << "[" << iheap << "][" << ichan << "][" << ipol << "][" << idat << "] = " << from[0] << "," << from[1] << endl;
              for (unsigned idim=0; idim<ndim; idim++)
              {
                if (nbit == 16)
                  into[idim] = float(from16[idim]);
                else
                  into[idim] = from32[idim];
              }
              into += ndim;
              from16 += ndim;  
              from32 += ndim;  
            }
          }
        }
      }
    }
    break;
    case TimeSeries::OrderTFP:
    {
      throw Error (InvalidState, "dsp::BoreasVoltageUnpacker::unpack",
                   "Unpack to OrderTFP not supported");
    }
    break;
    default:
      throw Error (InvalidState, "dsp::BoreasVoltageUnpacker::unpack",
                   "unrecognized output order");
    break;
  }
}
