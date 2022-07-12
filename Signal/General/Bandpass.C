/***************************************************************************
 *
 *   Copyright (C) 2002 - 2022 by Stephen Ord and Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/Bandpass.h"
#include "dsp/Apodization.h"
#include "dsp/Scratch.h"

#include "FTransform.h"

using namespace std;

dsp::Bandpass::Bandpass () :
  Transformation <TimeSeries, Response> ("Bandpass", outofplace) 
{
  resolution = 0;
  integration_length = 0;
  output_state = Signal::PPQQ;
  select_input_channel = -1;
}

dsp::Bandpass::~Bandpass ()
{
}

//! Set the apodization function
void dsp::Bandpass::set_apodization (Apodization* _function)
{
  apodization = _function; 
}

//! Set the frequency response function
void dsp::Bandpass::set_response (Response* _response)
{
  response = _response;
}

bool dsp::Bandpass::has_response () const
{
  return response;
}

const dsp::Response* dsp::Bandpass::get_response() const
{
  return response;
}

void dsp::Bandpass::transformation ()
{
  if (!resolution)
    throw Error (InvalidState, "dsp::Bandpass::transformation",
		 "number of output frequency channels == 0");

  unsigned npol = input->get_npol ();
  unsigned nchan = input->get_nchan ();
  unsigned ichan_offset = 0;
 
  if (verbose)
    cerr << "dsp::Bandpass::transformation input npol=" << npol
	 << " nchan=" << nchan << endl;
 
  if (select_input_channel > -1)
  {
    nchan = 1;
    ichan_offset = select_input_channel;
  }
   
  // 2 floats per complex number
  unsigned pts_reqd = resolution * 2;

  bool full_poln = npol == 2 &&
    (output_state == Signal::Stokes || output_state == Signal::Coherence);

  if (full_poln)
    // need space for one more complex spectrum
    pts_reqd += resolution * 2;

  // number of time samples in forward fft and overlap region
  unsigned nsamp_fft = 0;

  Signal::State state = input->get_state();

  if (state == Signal::Nyquist)
  {
    nsamp_fft = resolution * 2;
    pts_reqd += 4;
  }
  else if (state == Signal::Analytic)
  {
    nsamp_fft = resolution;
  }
  else
    {
      detected_input ();
      return;
    }

  unsigned output_ndat = output->get_ndat();

  if (full_poln)
    output->resize (4, nchan, resolution, 1);
  else
    output->resize (npol, nchan, resolution, 1);

  if (output_ndat == 0)
    output->zero();

  if (resolution == 1)
    detect_the_input ();

  // there must be at least enough data for one FFT
  if (input->get_ndat() < nsamp_fft)
    throw Error (InvalidState, "dsp::Bandpass::transformation",
		 "error ndat=" I64 " < nfft=%d", input->get_ndat(), nsamp_fft);

  // number of FFTs for this data block
  uint64_t npart = input->get_ndat() / nsamp_fft;

  if (verbose)
    cerr << "dsp::Bandpass::transformation npart=" << npart << endl;

  float* spectrum[2];
  spectrum[0] = scratch->space<float> (pts_reqd);
  spectrum[1] = spectrum[0];
  if (full_poln)
    spectrum[1] += resolution * 2;

  unsigned cross_pol = 1;
  if (full_poln)
    cross_pol = 2;

  // number of floats to step between each FFT
  unsigned step = resolution * 2;

  for (unsigned ichan=0; ichan < nchan; ichan++)
  {
    for (unsigned ipol=0; ipol < npol; ipol++)
    {
      for (uint64_t ipart=0; ipart < npart; ipart++)
      {
	uint64_t offset = ipart * step;
		
	for (unsigned jpol=0; jpol<cross_pol; jpol++)
	{
	  if (full_poln)
	    ipol = jpol;
	  
	  float* ptr = const_cast<float*>(input->get_datptr (ichan+ichan_offset, ipol));
	  ptr += offset;
	  
	  if (apodization)
	    apodization -> operate (ptr);

	  
	  if (state == Signal::Nyquist)
	    FTransform::frc1d (nsamp_fft, spectrum[ipol], ptr);

	  else if (state == Signal::Analytic)
	    FTransform::fcc1d (nsamp_fft, spectrum[ipol], ptr);
	  
	  // HERE is where the new fractional phase delay + fringe is applied
	  if (response)
	    response->operate (spectrum[ipol], ipol, ichan);
	}
	
	if (full_poln) 
	  output->integrate (spectrum[0], spectrum[1], ichan);

	else
	  output->integrate (spectrum[ipol], ipol, ichan);


      }  // for each part of the time series
    } // for each polarization
  } // for each frequency channel

  integration_length += double(npart*nsamp_fft) / input->get_rate();

  if ( input->get_dual_sideband() )
  {
    if (verbose)
      cerr << "dsp::Bandpass::transformation set swap" << endl;
    output->flagswap( nchan );
  }

  // for each poln
  // for each channel
}

void dsp::Bandpass::detected_input ()
{
  unsigned npol = input->get_npol ();
  unsigned nchan = input->get_nchan ();
  uint64_t ndat = input->get_ndat();

  output->resize (npol, 1, nchan, 1);
  reset_output();

  for (unsigned ipol=0; ipol < npol; ipol++)
  {
    float* result = output->get_datptr( 0, ipol );

    for (unsigned ichan=0; ichan < nchan; ichan++)
    {
      const float* dat = input->get_datptr( ichan, ipol );
      double tot = 0;
      for (uint64_t idat=0; idat < ndat; idat++)
	tot += dat[idat];

      result[ichan] += tot;
    }
  }

  integration_length += ndat / input->get_rate();
}

void dsp::Bandpass::detect_the_input ()
{
  unsigned npol = input->get_npol ();
  unsigned nchan = input->get_nchan ();
  uint64_t ndat = input->get_ndat();

  Signal::State state = input->get_state();

  // 2 floats per complex number
  unsigned pts_reqd = nchan * 2;

  bool full_poln = npol == 2 &&
    (output_state == Signal::Stokes || output_state == Signal::Coherence);

  if (full_poln)
    // need space for one more complex spectrum
    pts_reqd += nchan * 2;

  float* spectrum[2];
  spectrum[0] = scratch->space<float> (pts_reqd);
  spectrum[1] = spectrum[0];
  if (full_poln)
    spectrum[1] += nchan * 2;

  unsigned cross_pol = 1;
  if (full_poln)
    cross_pol = 2;

  // number of floats to step between each FFT
  unsigned step = 2;

  bool complex_valued = state == Signal::Analytic;

  for (uint64_t ipart=0; ipart < ndat; ipart++)
  {
    uint64_t offset = ipart * step;

    for (unsigned ipol=0; ipol < npol; ipol++)
    {
      for (unsigned jpol=0; jpol<cross_pol; jpol++)
      {
        if (full_poln)
          ipol = jpol;

        for (unsigned ichan=0; ichan < nchan; ichan++)
        {
          const float* ptr = input->get_datptr (ichan, ipol);
          ptr += offset;

          spectrum[ipol][ichan*2] = ptr[0];
          spectrum[ipol][ichan*2+1] = (complex_valued) ? ptr[1] : 0.0;
        }
      }

      if (full_poln)
        output->integrate (spectrum[0], spectrum[1]);
      else
        output->integrate (spectrum[ipol], ipol);

    } // for each polarization
  } // for each time sample

  integration_length += ndat / input->get_rate();
}

//! Set the integration length and bandpass to zero
void dsp::Bandpass::reset_output()
{
  if (verbose)
    cerr << "dsp::Bandpass::reset_output()" << endl;
  integration_length = 0;
  if (output)
    output -> zero();
}
