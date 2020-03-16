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

#include <errno.h>

using namespace std;

dsp::LFAASPEADUnpacker::LFAASPEADUnpacker (const char* _name) : HistUnpacker (_name)
{
  if (verbose)
    cerr << "dsp::LFAASPEADUnpacker ctor" << endl;

  set_nstate (256);
  table = new BitTable (8, BitTable::TwosComplement);
 
  npol = 2;
  ndim = 2;
}

dsp::LFAASPEADUnpacker::~LFAASPEADUnpacker ()
{
}

dsp::LFAASPEADUnpacker * dsp::LFAASPEADUnpacker::clone () const
{
  return new LFAASPEADUnpacker (*this);
}

bool dsp::LFAASPEADUnpacker::matches (const Observation* observation)
{
  return observation->get_machine() == "LFAASP"
    && observation->get_ndim() == 2
    && observation->get_npol() == 2
    && observation->get_nbit() == 8;
}

/*! The quadrature components are offset by one */
unsigned dsp::LFAASPEADUnpacker::get_output_offset (unsigned idig) const
{
  return idig % 2;
}

/*! The first two digitizer channels are poln0, the last two are poln1 */
unsigned dsp::LFAASPEADUnpacker::get_output_ipol (unsigned idig) const
{
  return (idig % 4) / 2;
}

/*! Each chan has 4 values (quadrature, dual pol) */
unsigned dsp::LFAASPEADUnpacker::get_output_ichan (unsigned idig) const
{
  return idig / 4;
}

void dsp::LFAASPEADUnpacker::unpack ()
{
  if (verbose)
    cerr << "dsp::LFAASPEADUnpacker::unpack()" << endl;

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
  const float* lookup = table->get_values ();

  // data is stored as sample blocks of FPT ordered data
  const uint64_t nval = nsamp_per_heap * ndim;

  if (verbose)
    cerr << "dsp::LFAASPEADUnpacker::unpack nheap=" << nheap << " ndat=" << ndat << " nchan=" << nchan
         << " npol=" << npol << " nval=" << nval << endl;

  unsigned long * digs[4];

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
