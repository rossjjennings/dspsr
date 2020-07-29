//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2020 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "dsp/LFAASPEADUnpacker.h"
#include "dsp/BitTable.h"

#include "Error.h"

#if HAVE_CUDA
#include "dsp/MemoryCUDA.h"
#include "dsp/LFAASPEADUnpackerCUDA.h"
#endif

//#include <errno.h>

using namespace std;

dsp::LFAASPEADUnpacker::LFAASPEADUnpacker (const char* _name) : HistUnpacker (_name)
{
  if (verbose)
    cerr << "dsp::LFAASPEADUnpacker ctor" << endl;

  set_nstate (256);
  set_ndig (2);
  table = new BitTable (8, BitTable::TwosComplement);

  npol = 2;
  ndim = 2;
  engine = NULL;
}

dsp::LFAASPEADUnpacker::~LFAASPEADUnpacker ()
{
}

dsp::LFAASPEADUnpacker * dsp::LFAASPEADUnpacker::clone () const
{
  return new LFAASPEADUnpacker (*this);
}

void dsp::LFAASPEADUnpacker::set_engine (Engine* _engine)
{
  cerr << "dsp::LFAASPEADUnpacker::set_engine" << endl;
  engine = _engine;
}

//! Return true if the unpacker can operate on the specified device
bool dsp::LFAASPEADUnpacker::get_device_supported (Memory* memory) const
{
  if (verbose)
    cerr << "dsp::LFAASPEADUnpacker::get_device_supported memory=" << (void *) memory << endl;
  bool supported = false;
#ifdef HAVE_CUDA
  supported = dynamic_cast< CUDA::DeviceMemory*> ( memory );
#endif
  if (verbose)
    cerr << "dsp::LFAASPEADUnpacker::get_device_supported supported=" << supported << endl;
  return supported;
}

//! Set the device on which the unpacker will operate
void dsp::LFAASPEADUnpacker::set_device (Memory* memory)
{
  if (verbose)
    cerr << "dsp::LFAASPEADUnpacker::set_device()" << endl;
#if HAVE_CUDA
  CUDA::DeviceMemory * gpu_mem = dynamic_cast< CUDA::DeviceMemory*>( memory );
  if (gpu_mem)
  {
    cudaStream_t stream = gpu_mem->get_stream();
    set_engine (new CUDA::LFAASPEADUnpackerEngine(stream));
  }
#endif

  if (engine)
    engine->setup ();
  else
    Unpacker::set_device (memory);

  device_prepared = true;
}

bool dsp::LFAASPEADUnpacker::matches (const Observation* observation)
{
  return observation->get_machine() == "LFAASP"
    && observation->get_ndim() == 2
    && observation->get_npol() == 2
    && observation->get_nbit() == 8;
}

unsigned dsp::LFAASPEADUnpacker::get_output_offset (unsigned idig) const
{
  return idig % 2;
}

unsigned dsp::LFAASPEADUnpacker::get_output_ipol (unsigned idig) const
{
  return (idig % 4) / 2;
}

unsigned dsp::LFAASPEADUnpacker::get_output_ichan (unsigned idig) const
{
  return idig / 4;
}

void dsp::LFAASPEADUnpacker::unpack ()
{
  // there are 4 digitisers per channel
  set_ndig (4 * input->get_nchan());

  // ensure the histograms are initialized
  unsigned long * digs[4];
  digs[0] = get_histogram (0);
  digs[1] = get_histogram (1);
  digs[2] = get_histogram (2);
  digs[3] = get_histogram (3);

  if (engine)
  {
    if (verbose)
      cerr << "dsp::LFAASPEADUnpacker::unpack using Engine" << endl;
    engine->unpack(table->get_scale(), input, output);
    return;
  }
  if (verbose)
    cerr << "dsp::LFAASPEADUnpacker::unpack using CPU" << endl;

  // some programs (digifil) do not call set_device
  if (! device_prepared )
    set_device ( Memory::get_manager ());

  if (output->get_order() != TimeSeries::OrderFPT)
    throw Error (InvalidState, "dsp::LFAASPEADUnpacker::unpack",
                 "cannot unpack into FPT order");

  int32_t * from = (int32_t *) input->get_rawptr();
  int32_t from32;
  int8_t * from8 = (int8_t * ) &from32;

  float * into_p0;
  float * into_p1;

  const float scale = table->get_scale();
  const uint64_t ndat = input->get_ndat();
  const unsigned nchan = input->get_nchan();
  const unsigned npol  = input->get_npol();
  const unsigned ndim  = 2;
  const unsigned nsamp_per_heap = 2048;
  const unsigned nheap = ndat / nsamp_per_heap;

  // data is stored as sample blocks of FPT ordered data
  const uint64_t nval = nsamp_per_heap * ndim;

  if (verbose)
    cerr << "dsp::LFAASPEADUnpacker::unpack nheap=" << nheap << " ndat=" << ndat << " nchan=" << nchan
         << " npol=" << npol << " nval=" << nval << endl;

  switch ( output->get_order() )
  {
    case TimeSeries::OrderFPT:
    {
#ifdef _DEBUG 
      cerr << "dsp::LFAASPEADUnpacker::unpack TimeSeries::OrderFPT" << endl;
#endif
      uint64_t heap_offset = 0;
      for (unsigned iheap=0; iheap<nheap; iheap++)
      {
        for (unsigned ichan=0; ichan<nchan; ichan++)
        {
          unsigned idig = ichan*ndim*npol;

          digs[0] = get_histogram (idig+0); 
          digs[1] = get_histogram (idig+1); 
          digs[2] = get_histogram (idig+2); 
          digs[3] = get_histogram (idig+3); 

          into_p0 = output->get_datptr (ichan, 0) + heap_offset;
          into_p1 = output->get_datptr (ichan, 1) + heap_offset;

          for (unsigned isamp=0; isamp<nsamp_per_heap; isamp++)
          {
            from32 = from[isamp];

            digs[0][(int) from8[0] + 128]++;
            digs[1][(int) from8[1] + 128]++;
            digs[2][(int) from8[2] + 128]++;
            digs[3][(int) from8[3] + 128]++;
                
            into_p0[(2*isamp) + 0] = float(from8[0]) * scale;
            into_p0[(2*isamp) + 1] = float(from8[1]) * scale;
            into_p1[(2*isamp) + 0] = float(from8[2]) * scale;
            into_p1[(2*isamp) + 1] = float(from8[3]) * scale;
          }

          from += nsamp_per_heap;
        }
	heap_offset += nsamp_per_heap * ndim;
      }
    }
    break;
    case TimeSeries::OrderTFP:
    {
#ifdef _DEBUG 
      cerr << "dsp::LFAASPEADUnpacker::unpack TimeSeries::OrderTFP" << endl;
#endif
    }
    break;
    default:
      throw Error (InvalidState, "dsp::LFAASPEADUnpacker::unpack",
                   "unrecognized output order");
    break;
  }
}
