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

#include "dsp/UWBFourBitUnpacker.h"
#include "dsp/ASCIIObservation.h"

#include "Error.h"

#include <errno.h>
#include <string.h>

using namespace std;

static void* const undefined_stream = (void *) -1;

dsp::UWBFourBitUnpacker::UWBFourBitUnpacker (const char* _name) : HistUnpacker (_name)
{
  if (verbose)
    cerr << "dsp::UWBFourBitUnpacker ctor" << endl;
 
  set_ndig (2); 
  set_nstate (16);

  npol = 2;
  ndim = 2;

  have_scales_and_offsets = false;
}

dsp::UWBFourBitUnpacker::~UWBFourBitUnpacker ()
{
}

unsigned dsp::UWBFourBitUnpacker::get_output_offset (unsigned idig) const
{
  return idig % 2;
}

unsigned dsp::UWBFourBitUnpacker::get_output_ipol (unsigned idig) const
{
  return idig / 2;
}

unsigned dsp::UWBFourBitUnpacker::get_output_ichan (unsigned idig) const
{
  return idig / 4;
}

unsigned dsp::UWBFourBitUnpacker::get_ndim_per_digitizer () const
{
  return 1;
}

dsp::UWBFourBitUnpacker * dsp::UWBFourBitUnpacker::clone () const
{
  return new UWBFourBitUnpacker (*this);
}

//! Return true if the unpacker support the specified output order
bool dsp::UWBFourBitUnpacker::get_order_supported (TimeSeries::Order order) const
{
  return (order == TimeSeries::OrderFPT);
}

//! Set the order of the dimensions in the output TimeSeries
void dsp::UWBFourBitUnpacker::set_output_order (TimeSeries::Order order)
{
  output_order = order;
}

bool dsp::UWBFourBitUnpacker::matches (const Observation* observation)
{
  if (!dynamic_cast<const ASCIIObservation *>(observation))
  {
    if (verbose)
      cerr << "dsp::UWBFourBitUnpacker::matches"
              " ASCIIObservation required and not available" << endl;
    return false;
  }
  
  return (observation->get_machine()== "UWB" || observation->get_machine()== "Medusa")
    && observation->get_nchan() == 1
    && observation->get_ndim() == 2
    && (observation->get_npol() == 2 || observation->get_npol() == 1)
    && observation->get_nbit() == 4;
}

void dsp::UWBFourBitUnpacker::get_scales_and_offsets ()
{
  unsigned nchan = input->get_nchan();
  unsigned npol = input->get_npol();

  const Input * in = input->get_loader();
  const Observation * obs = in->get_info();
  const ASCIIObservation * info = dynamic_cast<const ASCIIObservation *>(obs);
  if (!info)
    throw Error (InvalidState, "dsp::UWBFourBitUnpacker::get_scales_and_offsets",
                 "ASCIIObservation required and not available");

  if (verbose)
    cerr << "dsp::UWBFourBitUnpacker::get_scales_and_offsets nchan=" << nchan << " npol=" << npol << endl;

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
      {
        cerr << "scales[" << ichan << "][" << ipol << "]=" << scales[ichan][ipol] << endl;
        cerr << "offsets[" << ichan << "][" << ipol << "]=" << offsets[ichan][ipol] << endl;
      }
    }
  }
}

void dsp::UWBFourBitUnpacker::unpack ()
{
  if (verbose)
    cerr << "dsp::UWBFourBitUnpacker::unpack()" << endl;

  npol = input->get_npol();
  set_ndig (npol*2);

  if (!have_scales_and_offsets)
  {
    if (verbose)
      cerr << "dsp::UWBFourBitUnpacker::unpack getting scales and offsets" << endl;
    get_scales_and_offsets();
    have_scales_and_offsets = true;
  }


  // Data are stored in TFP order, but nchan == 1, so TP order
  unsigned ichan = 0;
  unsigned long * hists[2];

  const uint64_t ndat = input->get_ndat();
  const unsigned into_stride = ndim;
  const unsigned from_stride = npol;

  if (verbose)
    cerr << "dsp::UWBFourBitUnpacker::unpack ndat=" << ndat 
         << " ndim=" << ndim << " npol=" << npol << endl;

  for (unsigned ipol=0; ipol<npol; ipol++)
  {
    char * from = ((char *) input->get_rawptr()) + ipol;
    float * into = output->get_datptr (ichan, ipol);
    hists[0] = get_histogram (ipol*2 + 0);
    hists[1] = get_histogram (ipol*2 + 1);

    if (verbose)
      cerr << "dsp::UWBFourBitUnpacker::unpack unpacking ipol=" << ipol << endl;

    const float offset = offsets[ichan][ipol];
    const float scale = scales[ichan][ipol];

    for (unsigned idat=0; idat<ndat; idat++)
    {
      char packed = (char) (*from);

      int8_t real = int8_t((packed & 0x0f) << 4) / 16;
      int8_t imag = int8_t( packed & 0xf0)       / 16;

      into[0] = (float(real) * scale) + offset;
      into[1] = (float(imag) * scale) + offset;

      real = max(int8_t(-8), real);
      real = min(int8_t(7), real);
      imag = max(int8_t(-8), imag);
      imag = min(int8_t(7), imag);
      hists[0][real+8]++;
      hists[1][imag+8]++;

      into += into_stride; 
      from += from_stride; 

    } // for each complex sample
    if (verbose)
      cerr << "dsp::UWBFourBitUnpacker::unpack finished unpacking ipol=" << ipol << endl;
  } // for each polarisation
}

