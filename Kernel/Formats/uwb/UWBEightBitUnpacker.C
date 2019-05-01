//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2019 by Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "dsp/UWBEightBitUnpacker.h"
#include "dsp/ASCIIObservation.h"

#include "Error.h"

#include <errno.h>
#include <string.h>

using namespace std;

static void* const undefined_stream = (void *) -1;

dsp::UWBEightBitUnpacker::UWBEightBitUnpacker (const char* _name) : HistUnpacker (_name)
{
  if (verbose)
    cerr << "dsp::UWBEightBitUnpacker ctor" << endl;
 
  set_ndig (2); 
  set_nstate (256);

  npol = 2;
  ndim = 2;

  have_scales_and_offsets = false;
}

dsp::UWBEightBitUnpacker::~UWBEightBitUnpacker ()
{
}

unsigned dsp::UWBEightBitUnpacker::get_output_offset (unsigned idig) const
{
  return idig % 2;
}

unsigned dsp::UWBEightBitUnpacker::get_output_ipol (unsigned idig) const
{
  return idig / 2;
}

unsigned dsp::UWBEightBitUnpacker::get_output_ichan (unsigned idig) const
{
  return idig / 4;
}

unsigned dsp::UWBEightBitUnpacker::get_ndim_per_digitizer () const
{
  return 1;
}

dsp::UWBEightBitUnpacker * dsp::UWBEightBitUnpacker::clone () const
{
  return new UWBEightBitUnpacker (*this);
}

//! Return true if the unpacker support the specified output order
bool dsp::UWBEightBitUnpacker::get_order_supported (TimeSeries::Order order) const
{
  return (order == TimeSeries::OrderFPT);
}

//! Set the order of the dimensions in the output TimeSeries
void dsp::UWBEightBitUnpacker::set_output_order (TimeSeries::Order order)
{
  output_order = order;
}

bool dsp::UWBEightBitUnpacker::matches (const Observation* observation)
{
  if (!dynamic_cast<const ASCIIObservation *>(observation))
  {
    if (verbose)
      cerr << "dsp::UWBEightBitUnpacker::matches"
              " ASCIIObservation required and not available" << endl;
    return false;
  }
  
  return (observation->get_machine()== "UWB" || observation->get_machine()== "Medusa")
    && observation->get_nchan() == 1
    && observation->get_ndim() == 2
    && (observation->get_npol() == 2 || observation->get_npol() == 1)
    && observation->get_nbit() == 8;
}

void dsp::UWBEightBitUnpacker::get_scales_and_offsets ()
{
  unsigned nchan = input->get_nchan();
  unsigned npol = input->get_npol();

  const Input * in = input->get_loader();
  const Observation * obs = in->get_info();
  const ASCIIObservation * info = dynamic_cast<const ASCIIObservation *>(obs);
  if (!info)
    throw Error (InvalidState, "dsp::UWBEightBitUnpacker::get_scales_and_offsets",
                 "ASCIIObservation required and not available");

  scales.resize(nchan);
  offsets.resize(nchan);
  stringstream key;
  for (unsigned ichan=0; ichan<nchan; ichan++)
  {
    offsets[ichan].resize(npol);
    scales[ichan].resize(npol);
    for (unsigned ipol=0; ipol<npol; ipol++)
    {
      key.str("");
      key << "DAT_SCL_" << ichan << "_" << ipol;
      try
      {
        info->custom_header_get (key.str().c_str(), "%f", &(scales[ichan][ipol]));
      }
      catch (Error& error) 
      {
        scales[ichan][ipol] = 1.0f;
      }

      key.str("");
      key << "DAT_OFF_" << ichan << "_" << ipol;
      try
      {
        info->custom_header_get (key.str().c_str(), "%f", &(offsets[ichan][ipol]));
      }
      catch (Error& error)
      {
        offsets[ichan][ipol] = 0;
      }

      if (verbose)
        cerr << "dsp::UWBEightBitUnpacker::get_scales_and_offsets ichan=" 
             << ichan << " ipol=" << ipol << " offset=" << offsets[ichan][ipol]
             << " scale="<< scales[ichan][ipol] << endl;
    }
  }
}

void dsp::UWBEightBitUnpacker::unpack ()
{
  if (verbose)
    cerr << "dsp::UWBEightBitUnpacker::unpack()" << endl;

  npol = input->get_npol();
  set_ndig (npol*2);

  if (!have_scales_and_offsets)
  {
    if (verbose)
      cerr << "dsp::UWBEightBitUnpacker::unpack reading scales and offsets" << endl;
    get_scales_and_offsets();
    have_scales_and_offsets = true;
  }

  // Data are stored in TFP order, but nchan == 1, so TP order
  unsigned ichan = 0;
  unsigned long * hists[2];

  const uint64_t ndat = input->get_ndat();
  const unsigned into_stride = ndim;
  const unsigned from_stride = npol * ndim;

  if (verbose)
    cerr << "dsp::UWBEightBitUnpacker::unpack ndat=" << ndat 
         << " ndim=" << ndim << " npol=" << npol << endl;

  for (unsigned ipol=0; ipol<npol; ipol++)
  {
    int8_t * from = ((int8_t *) input->get_rawptr()) + (ipol*ndim);
    float * into = output->get_datptr (ichan, ipol);
    hists[0] = get_histogram (ipol*2 + 0);
    hists[1] = get_histogram (ipol*2 + 1);

    if (verbose)
      cerr << "dsp::UWBEightBitUnpacker::unpack unpacking ipol=" << ipol << endl;

    const float offset = offsets[ichan][ipol];
    const float scale = scales[ichan][ipol];

    for (unsigned idat=0; idat<ndat; idat++)
    {
      const int real = from[0];
      const int imag = from[1];

      into[0] = (float(real) * scale) + offset;
      into[1] = (float(imag) * scale) + offset;

      hists[0][real+128]++;
      hists[1][imag+128]++;

      into += into_stride; 
      from += from_stride; 

    } // for each complex sample
  } // for each polarisation
}
