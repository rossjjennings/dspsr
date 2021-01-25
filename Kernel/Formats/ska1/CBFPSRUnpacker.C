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

#include "dsp/CBFPSRUnpacker.h"

#include "Error.h"

//#include <errno.h>

using namespace std;

dsp::CBFPSRUnpacker::CBFPSRUnpacker (const char* _name) : HistUnpacker (_name)
{
  if (verbose)
    cerr << "dsp::CBFPSRUnpacker ctor" << endl;

  set_nstate (65536);
  set_ndig (2);

  npol = 2;
  ndim = 2;
}

dsp::CBFPSRUnpacker::~CBFPSRUnpacker ()
{
}

dsp::CBFPSRUnpacker * dsp::CBFPSRUnpacker::clone () const
{
  return new CBFPSRUnpacker (*this);
}

bool dsp::CBFPSRUnpacker::matches (const Observation* observation)
{
  return observation->get_machine() == "LowCBF"
    && observation->get_ndim() == 2
    && observation->get_npol() == 2
    && (observation->get_nbit() == 8 || observation->get_nbit() == 16);
}

unsigned dsp::CBFPSRUnpacker::get_output_offset (unsigned idig) const
{
  return idig % 2;
}

unsigned dsp::CBFPSRUnpacker::get_output_ipol (unsigned idig) const
{
  return (idig % 4) / 2;
}

unsigned dsp::CBFPSRUnpacker::get_output_ichan (unsigned idig) const
{
  return idig / 4;
}

void dsp::CBFPSRUnpacker::unpack ()
{
  // there are 4 digitisers per channel
  set_ndig (4 * input->get_nchan());

  // ensure the histograms are initialized
  unsigned long * digs[4];
  digs[0] = get_histogram (0);
  digs[1] = get_histogram (1);
  digs[2] = get_histogram (2);
  digs[3] = get_histogram (3);

  if (output->get_order() != TimeSeries::OrderFPT)
    throw Error (InvalidState, "dsp::CBFPSRUnpacker::unpack",
                 "cannot unpack into FPT order");

  int32_t * from = (int32_t *) input->get_rawptr();
  int32_t from32;
  int16_t * from16 = (int16_t * ) &from32;
  float * into;

  const uint64_t ndat = input->get_ndat();
  const unsigned nchan = input->get_nchan();
  const unsigned npol  = input->get_npol();
  const unsigned ndim  = 2;
  const unsigned nsamp_per_heap = 32;
  const unsigned nheap = ndat / nsamp_per_heap;

  // data is stored as sample blocks of FPT ordered data
  const uint64_t nval = nsamp_per_heap * ndim;

  if (verbose)
    cerr << "dsp::CBFPSRUnpacker::unpack nheap=" << nheap << " ndat=" << ndat << " nchan=" << nchan
         << " npol=" << npol << " nval=" << nval << endl;

  switch ( output->get_order() )
  {
    case TimeSeries::OrderFPT:
    {
#ifdef _DEBUG 
      cerr << "dsp::CBFPSRUnpacker::unpack TimeSeries::OrderFPT" << endl;
#endif
      uint64_t heap_offset = 0;
      for (unsigned iheap=0; iheap<nheap; iheap++)
      {
        for (unsigned ichan=0; ichan<nchan; ichan++)
        {
          for (unsigned ipol=0; ipol<npol; ipol++)
          {
            into = output->get_datptr (ichan, ipol) + heap_offset;
            unsigned idig = ichan*ndim*npol + (ndim * ipol);
            digs[0] = get_histogram (idig+0); 
            digs[1] = get_histogram (idig+1); 

            for (unsigned isamp=0; isamp<nsamp_per_heap; isamp++)
            {
              from32 = from[isamp];

              digs[0][(int) from16[0] + 32768]++;
              digs[1][(int) from16[1] + 32768]++;
                
              into[(2*isamp) + 0] = float(from16[0]);
              into[(2*isamp) + 1] = float(from16[1]);
            }
            from += nsamp_per_heap;
          }
        }
        heap_offset += nsamp_per_heap * ndim;
      }
    }
    break;
    case TimeSeries::OrderTFP:
    {
#ifdef _DEBUG 
      cerr << "dsp::CBFPSRUnpacker::unpack TimeSeries::OrderTFP" << endl;
#endif
    }
    break;
    default:
      throw Error (InvalidState, "dsp::CBFPSRUnpacker::unpack",
                   "unrecognized output order");
    break;
  }
}
