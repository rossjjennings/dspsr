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

#include "dsp/BoreasSurveyUnpacker.h"

#include "Error.h"

using namespace std;

dsp::BoreasSurveyUnpacker::BoreasSurveyUnpacker (const char* _name) : HistUnpacker (_name)
{
  if (verbose)
    cerr << "dsp::BoreasSurveyUnpacker ctor" << endl;

  set_nstate (256);
  set_ndig (2);

  npol = 4;
  ndim = 1;
}

dsp::BoreasSurveyUnpacker::~BoreasSurveyUnpacker ()
{
}

dsp::BoreasSurveyUnpacker * dsp::BoreasSurveyUnpacker::clone () const
{
  return new BoreasSurveyUnpacker (*this);
}

bool dsp::BoreasSurveyUnpacker::matches (const Observation* observation)
{
  return observation->get_machine() == "Boreas"
    && observation->get_ndim() == 1
    && observation->get_npol() == 4
    && observation->get_nbit() == 8;
}

unsigned dsp::BoreasSurveyUnpacker::get_output_offset (unsigned idig) const
{
  return idig % 2;
}

unsigned dsp::BoreasSurveyUnpacker::get_output_ipol (unsigned idig) const
{
  return (idig % 4) / 2;
}

unsigned dsp::BoreasSurveyUnpacker::get_output_ichan (unsigned idig) const
{
  return idig / 4;
}

void dsp::BoreasSurveyUnpacker::unpack ()
{
  // there are 4 digitisers per channel
  set_ndig (4 * input->get_nchan());

  const int8_t * from = (int8_t *) input->get_rawptr();
  float * into;

  const uint64_t ndat = input->get_ndat();
  const unsigned nchan = input->get_nchan();
  const unsigned npol  = input->get_npol();
  const unsigned ndim  = input->get_ndim();
  const unsigned nsamp_per_heap = 1728;
  const unsigned nheap = ndat / nsamp_per_heap;

  // data is stored as sample blocks of FPT ordered data
  const uint64_t nval = nsamp_per_heap * ndim;

  if (verbose)
    cerr << "dsp::BoreasSurveyUnpacker::unpack nheap=" << nheap << " ndat=" << ndat << " nchan=" << nchan
         << " npol=" << npol << " nval=" << nval << " ndim=" << ndim << endl;

  // input are ordered in heaps of 1728 samples, packed in FPT order
  switch ( output->get_order() )
  {
    case TimeSeries::OrderFPT:
    {
#ifdef _DEBUG 
      cerr << "dsp::BoreasSurveyUnpacker::unpack TimeSeries::OrderFPT" << endl;
#endif
      uint64_t heap_offset = 0;
      for (unsigned iheap=0; iheap<nheap; iheap++)
      {
        unsigned ihist = 0;
        for (unsigned ichan=0; ichan<nchan; ichan++)
        {
          for (unsigned ipol=0; ipol<npol; ipol++)
          {
            into = output->get_datptr (ichan, ipol) + heap_offset;
            unsigned long * hist = get_histogram(ihist);

            for (unsigned isamp=0; isamp<nsamp_per_heap; isamp++)
            {
              const int packed = int(from[isamp]);
              hist[packed]++;
              into[isamp] = float(packed);
            }

            from += nsamp_per_heap;
            ihist++;
          }
        }
        heap_offset += nsamp_per_heap;
      }
    }
    break;
    case TimeSeries::OrderTFP:
    {
#ifdef _DEBUG 
      cerr << "dsp::BoreasSurveyUnpacker::unpack TimeSeries::OrderTFP" << endl;
#endif
      float * into = output->get_dattfp();

      const unsigned pol_stride = 1;
      const unsigned chan_stride = npol * pol_stride;
      const unsigned dat_stride = nchan * chan_stride;
      const unsigned heap_stride = nsamp_per_heap * dat_stride;
      
      for (unsigned iheap=0; iheap<nheap; iheap++)
      {
        const unsigned heap_offset = iheap * heap_stride;
        unsigned ihist = 0;
        for (unsigned ichan=0; ichan<nchan; ichan++)
        {
          const unsigned chan_offset = heap_offset + (ichan * chan_stride);
          for (unsigned ipol=0; ipol<npol; ipol++)
          {
            unsigned odx = chan_offset + ipol;
            unsigned long * hist = get_histogram(ihist);
            for (unsigned isamp=0; isamp<nsamp_per_heap; isamp++)
            {
              const int packed = int(from[isamp]);
              hist[packed]++;
              into[odx] = float(packed);
              odx += dat_stride;
            }

            from += nsamp_per_heap;
            ihist++;
          }
        }
      }
    }
    break;
    default:
      throw Error (InvalidState, "dsp::BoreasSurveyUnpacker::unpack",
                   "unrecognized output order");
    break;
  }
}
