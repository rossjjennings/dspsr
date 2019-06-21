/***************************************************************************
 *
 *   Copyright (C) 2019 by Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/DelayStartTime.h"

#include <assert.h>

using namespace std;

dsp::DelayStartTime::DelayStartTime ()
  : Transformation<TimeSeries,TimeSeries> ("DelayStartTime", inplace)
{
  start_mjd = MJD::zero;
  delay_samples = 0;
  delay_applied = false;
}

void dsp::DelayStartTime::set_engine (Engine* _engine)
{
  engine = _engine;
}

void dsp::DelayStartTime::set_start_time (MJD mjd)
{
  start_mjd = mjd;
}

//! prepare the transformation
void dsp::DelayStartTime::prepare ()
{
  if (verbose)
    cerr << "dsp::DelayStartTime::prepare start_mjd=" << start_mjd<< endl;

  prepare_output ();
}

//! prepare the output timeseries
void dsp::DelayStartTime::prepare_output ()
{
  const uint64_t ndat  = input->get_ndat();
  int64_t input_sample = input->get_input_sample();
  MJD input_start_time = input->get_start_time();

  uint64_t output_ndat = 0;

  if (!delay_applied)
  {
    if (verbose)
      cerr << "dsp::DelayStartTime::prepare_output calculating delay for MJD "
           << start_mjd << endl;

    // if the start_mjd has not been configured
    if (start_mjd == MJD::zero)
    {
      delay_applied = true;
      delay_samples = 0;
    }
    // if the input is not yet configured
    else if ((input_start_time == MJD::zero) || (input_sample == -1))
    {
      if (verbose)
        cerr << "dsp::DelayStartTime::prepare_output input start time not "
             << "known" << endl;
      delay_samples = 0;
    }
    else
    {
      // the delay between the nominated start time and start of data in this block
      MJD block_delay = start_mjd - input_start_time;

      double block_delay_seconds = block_delay.in_seconds();
      uint64_t block_delay_samples = int64_t(block_delay_seconds * input->get_rate()) + 1;

      delay_samples = block_delay_samples + input_sample;
      if (verbose)
        cerr << "dsp::DelayStartTime::prepare_output start_mjd=" << start_mjd 
             << " start_time=" << input_start_time
             << " block_delay=" << block_delay.in_seconds() << " seconds" 
             << " or " << block_delay_samples << " samples delay_samples="
             << delay_samples << endl;

      if ((input_sample + ndat) < delay_samples)
      {
        output_ndat = 0;
      }
      else
      {
        output_ndat = ndat - block_delay_samples; 
      }
    } 
  }
  else
    output_ndat = ndat;

  if (verbose)
    cerr << "dsp::DelayStartTime::prepare_output input_ndat=" << ndat
         << " output_ndat=" << output_ndat << " delay=" << delay_samples
         << endl;

  // prepare the output timeseries
  get_output()->copy_configuration (get_input());

  // set the number of output samples
  output->set_ndat (output_ndat);

  // adjust the start time by the number of samples, until the
  // delay has been applied
  if (!delay_applied)
  {
    output->change_start_time (delay_samples);
  }

  // set the output sample number
  uint64_t output_sample = 0;
  if (input_sample > delay_samples)
  {
    output_sample = input_sample - delay_samples;
    if (verbose)
      cerr << "dsp::DelayStartTime::prepare_output input_sample=" << input_sample
           << " delay_samples=" << delay_samples << " output_sample="
           << output_sample << endl;
  }
  output->set_input_sample (output_sample);
}

void dsp::DelayStartTime::transformation ()
{
  if (verbose)
    cerr << "dsp::DelayStartTime::transformation" << endl;

  prepare ();

  // the delay to the start time has already been performed
  if (delay_applied)
  {
    if (verbose)
      cerr << "dsp::DelayStartTime::transformation delay already applied" << endl;
    return;
  }

  const uint64_t input_ndat  = input->get_ndat();
  uint64_t output_ndat = output->get_ndat();

  if (verbose)
    cerr << "dsp::DelayStartTime::transformation input_ndat=" << input_ndat
         << " output_ndat=" << output_ndat << endl;

  // only transform the data if we are applying the delay in this block
  if (output_ndat == 0)
  {
    if (verbose)
      cerr << "dsp::DelayStartTime::transformation output_ndat==0, skipping transformation" << endl;
    return;
  }

  if (engine)
  {
    if (verbose)
      cerr << "dsp::DelayStartTime::transformation engine->delay ()" << endl;
    engine->delay (input, output, output_ndat, delay_samples);
  }
  else
  {
    const unsigned ndim  = input->get_ndim();
    const unsigned npol  = input->get_npol();
    const unsigned nchan = input->get_nchan();

    uint64_t output_nfloat = output_ndat * ndim;
    for (unsigned ipol=0; ipol < npol; ipol++)
    {
      for (unsigned ichan=0; ichan < nchan; ichan++)
      {
        const float* in_data = input->get_datptr (ichan, ipol);
        in_data += delay_samples * ndim;
        float* out_data = output->get_datptr (ichan, ipol);

        for (uint64_t idat=0; idat < output_nfloat; idat++)
          out_data[idat] = in_data[idat];
      }
    }
  }

  // mark that the delay has been applied
  delay_applied = true;
}
