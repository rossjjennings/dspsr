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

#include "dsp/AAVS2Unpacker.h"
#include "dsp/BitTable.h"

#include "Error.h"

#if HAVE_CUDA
#include "dsp/MemoryCUDA.h"
#include "dsp/AAVS2UnpackerCUDA.h"
#endif

//#include <errno.h>

using namespace std;

static void* const undefined_stream = (void *) -1;

dsp::AAVS2Unpacker::AAVS2Unpacker (const char* _name) : HistUnpacker (_name)
{
  if (verbose)
    cerr << "dsp::AASVS2Unpacker ctor" << endl;

  set_nstate (256);
  table = new BitTable (8, BitTable::TwosComplement);

  set_ndig (2);
  npol = 2;
  ndim = 2;
  engine = NULL;
}

dsp::AAVS2Unpacker::~AAVS2Unpacker ()
{
}

dsp::AAVS2Unpacker * dsp::AAVS2Unpacker::clone () const
{
  return new AAVS2Unpacker (*this);
}

void dsp::AAVS2Unpacker::set_engine (Engine* _engine)
{
  cerr << "dsp::AAVS2Unpacker::set_engine" << endl;
  engine = _engine;
}

//! Return true if the unpacker can operate on the specified device
bool dsp::AAVS2Unpacker::get_device_supported (Memory* memory) const
{
  if (verbose)
    cerr << "dsp::AAVS2Unpacker::get_device_supported memory=" << (void *) memory << endl;
  bool supported = false;
#ifdef HAVE_CUDA
  supported = dynamic_cast< CUDA::DeviceMemory*> ( memory );
#endif
  if (verbose)
    cerr << "dsp::AAVS2Unpacker::get_device_supported supported=" << supported << endl;
  return supported;
}

//! Set the device on which the unpacker will operate
void dsp::AAVS2Unpacker::set_device (Memory* memory)
{
  if (verbose)
    cerr << "dsp::AAVS2Unpacker::set_device()" << endl;
#if HAVE_CUDA
  CUDA::DeviceMemory * gpu_mem = dynamic_cast< CUDA::DeviceMemory*>( memory );
  if (gpu_mem)
  {
    cudaStream_t stream = gpu_mem->get_stream();
    set_engine (new CUDA::AAVS2UnpackerEngine(stream));
  }
#endif

  if (engine)
    engine->setup ();
  else
    Unpacker::set_device (memory);

  device_prepared = true;
}

bool dsp::AAVS2Unpacker::matches (const Observation* observation)
{
  return observation->get_machine()== "AAVS2"
    && observation->get_nchan() == 1
    && observation->get_ndim() == 2
    && observation->get_npol() == 2
    && observation->get_nbit() == 8;
}

/*! The first two digitizer channels are poln0, the last two are poln1 */
unsigned dsp::AAVS2Unpacker::get_output_ipol (unsigned idig) const
{
  return idig % 2;
}

/*! Each chan has 2 values (dual pol) */
unsigned dsp::AAVS2Unpacker::get_output_ichan (unsigned idig) const
{
  return idig / 2;
}

/*! Number of dimensions per digitizer */
unsigned dsp::AAVS2Unpacker::get_ndim_per_digitizer () const
{
  return 2;
}

void dsp::AAVS2Unpacker::unpack ()
{
  // ensure the histograms are initialized
  unsigned long * digs[2];
  digs[0] = get_histogram (0);
  digs[1] = get_histogram (1);

  if (engine)
  {
    if (verbose)
      cerr << "dsp::AAVS2Unpacker::unpack using Engine" << endl;
    engine->unpack(table->get_scale(), input, output);
    return;
  }
  if (verbose)
    cerr << "dsp::AAVS2Unpacker::unpack using CPU" << endl;

  // some programs (digifil) do not call set_device
  if (! device_prepared )
    set_device ( Memory::get_manager ());

  if (output->get_order() != TimeSeries::OrderFPT)
    throw Error (InvalidState, "dsp::AAVS2Unpacker::unpack",
                 "cannot unpack into FPT order");

  // Data format is TFP
  const uint64_t ndat   = input->get_ndat();
  const unsigned nchan  = input->get_nchan();

  unsigned in_offset = 0;
  const unsigned into_stride = ndim;
  const unsigned from_stride = nchan * ndim * npol;
  const float scale = table->get_scale();

  for (unsigned ichan=0; ichan<nchan; ichan++)
  {
    // the 2 digitisers for this channel
    unsigned idig = ichan*npol;
    digs[0] = get_histogram (idig+0);
    digs[1] = get_histogram (idig+1);

    for (unsigned ipol=0; ipol<npol; ipol++)
    {
      float * into = output->get_datptr (ichan, ipol);
      const int8_t * from = (int8_t *) input->get_rawptr() + in_offset;

      for (uint64_t idat=0; idat<ndat; idat++)
      {
        int from0 = (int) from[0];
        int from1 = (int) from[1];
 
        if (from0 == -128)
          from0 = 0;
        if (from1 == -127)
          from1 = 0;
        into[0] = float(from0) * scale;
        into[1] = float(from1) * scale;

        digs[ipol][from0 + 128]++;
        digs[ipol][from1 + 128]++;

        into += into_stride;
        from += from_stride;
      }
      in_offset += ndim;
    }
  }
}
