/***************************************************************************
 *
 *   Copyright (C) 2006 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/SampleDelay.h"
#include "dsp/SampleDelayFunction.h"
#include "dsp/InputBuffering.h"

// #define _DEBUG 1

#include <assert.h>

using namespace std;

dsp::SampleDelay::SampleDelay ()
  : Transformation<TimeSeries,TimeSeries> ("SampleDelay", anyplace)
{
  zero_delay = 0;
  total_delay = 0;
  delay_span = 0;
  built = false;
  engine = NULL;

  set_buffering_policy (new InputBuffering (this));
}

void dsp::SampleDelay::set_engine (Engine* _engine)
{
  engine = _engine;
}

uint64_t dsp::SampleDelay::get_total_delay () const
{
  if (!built)
    const_cast<SampleDelay*>(this)->build();
  return total_delay;
}

int64_t dsp::SampleDelay::get_zero_delay () const
{
  if (!built)
    const_cast<SampleDelay*>(this)->build();
  return zero_delay;
}

//! Set the delay function
void dsp::SampleDelay::set_function (SampleDelayFunction* f)
{
  if (function && f == function.get())
    return;

  function = f;
  built = false;
}

void dsp::SampleDelay::set_delay_span (unsigned _delay_span)
{
  delay_span = _delay_span;
}

void dsp::SampleDelay::build ()
{
  if (verbose)
    cerr << "dsp::SampleDelay::build" << endl;

  unsigned input_npol  = input->get_npol();
  unsigned input_nchan = input->get_nchan();

  zero_delays.resize(input_nchan);
  fill(zero_delays.begin(), zero_delays.end(), 0);

  if (function->get_absolute())
  {
    zero_delay = 0;

    total_delay = function->get_delay (0, 0);

    for (unsigned ipol=0; ipol < input_npol; ipol++)
      for (unsigned ichan=0; ichan < input_nchan; ichan++)
        if (function->get_delay (ichan, ipol) > total_delay)
          total_delay = function->get_delay (ichan, ipol);

    return;
  }

  if (delay_span == 0)
    delay_span = input_nchan;

  for (unsigned ichan=0; ichan < input_nchan; ichan += delay_span)
  {
    // delay at the centre of the span
    unsigned from_chan = ichan;
    unsigned to_chan = ichan + (delay_span - 1);
    int64_t span_centre_delay = function->get_delay_range (from_chan, to_chan, 0);

    // compute the delay in each channel relative to the span_centre_delay
    for (unsigned ipol=0; ipol < input_npol; ipol++)
    {
      for (unsigned i=0; i<delay_span; i++)
      {
        zero_delays[ichan + i] = span_centre_delay;
        int64_t local_delay = function->get_delay (ichan + i, ipol) - span_centre_delay;
        if (local_delay > zero_delay)
          zero_delay = local_delay;
      }
    }
  }

  for (unsigned ichan=0; ichan<input_nchan; ichan++)
    zero_delays[ichan] += zero_delay;

  if (verbose)
    cerr << "dsp::SampleDelay::build zero delay = " << zero_delay
         << " samples" << endl;

  total_delay = 0;

  for (unsigned ipol=0; ipol < input_npol; ipol++)
  {
    for (unsigned ichan=0; ichan < input_nchan; ichan++)
    {
      int64_t relative_delay = int64_t(zero_delays[ichan]) - function->get_delay(ichan, ipol);
      if (relative_delay > int64_t(total_delay))
        total_delay = uint64_t(relative_delay);
    }
  }

  if (verbose)
    cerr << "dsp::SampleDelay::build total delay = " << total_delay
         << " samples" << endl;

  if (engine)
  {
    if (verbose)
      cerr << "dsp::SampleDelay::build engine->set_delays(" << input_npol
           << "," << input_nchan << ", zero_delays, function)" << endl;
    engine->set_delays(input_npol, input_nchan, zero_delays, function);
  }

  built = true;
}

//! prepare the transformation
void dsp::SampleDelay::prepare ()
{
  if (function->match(input) || !built)
    build ();

  if (!has_buffering_policy())
    return;

  if (verbose)
    cerr << "dsp::SampleDelay::prepare reserve=" << total_delay << endl;

  get_buffering_policy()->set_minimum_samples (total_delay);
}

//! prepare the output timeseries
void dsp::SampleDelay::prepare_output (uint64_t output_ndat)
{
  // prepare the output timeseries
  if (verbose)
    cerr << "dsp::SampleDelay::prepare_output output->copy_configuration(input)" << endl;
  get_output()->copy_configuration (get_input());

  if (output != input)
  {
    output->resize (output_ndat);
    get_output()->set_input_sample (get_input()->get_input_sample());
  }
  else
  {
    output->set_ndat (output_ndat);
  }

  // zero_delay
  output->change_start_time (zero_delay);
}

/*!
  \pre input TimeSeries must contain complex (Analytic) data
*/
void dsp::SampleDelay::transformation ()
{
  if (verbose)
    cerr << "dsp::SampleDelay::transformation" << endl;

  prepare ();

  const uint64_t input_ndat  = input->get_ndat();
  const unsigned input_ndim  = input->get_ndim();
  const unsigned input_npol  = input->get_npol();
  const unsigned input_nchan = input->get_nchan();

  uint64_t output_ndat = 0;

  if (input_ndat < total_delay)
  {
    if (verbose)
      cerr << "dsp::SampleDelay::transformation insufficient data\n"
        "  input ndat=" << input_ndat << " total delay=" << total_delay  << endl;
  }
  else
    output_ndat = input_ndat - total_delay;

  if (verbose)
    cerr << "dsp::SampleDelay::transformation set_next_start(" << output_ndat << ")" << endl;
  get_buffering_policy()->set_next_start (output_ndat);

  //prepare the output TimeSeries
  prepare_output (output_ndat);

  if (!output_ndat)
    return;

  if (engine)
  {
    if (verbose)
      cerr << "dsp::SampleDelay::transformation engine->retard (input, output, " << output_ndat << ")" << endl;
    engine->retard (input, output, output_ndat);
  }
  else
  {
    uint64_t output_nfloat = output_ndat * input_ndim;

    for (unsigned ipol=0; ipol < input_npol; ipol++) {

      for (unsigned ichan=0; ichan < input_nchan; ichan++) {

        const float* in_data = input->get_datptr (ichan, ipol);

        int64_t applied_delay = 0;

        if (zero_delay)
          // delays are relative to maximum delay
          applied_delay = zero_delays[ichan] - function->get_delay(ichan, ipol);
        else
          // delays are absolute and guaranteed positive
          applied_delay = function->get_delay(ichan, ipol);

        assert (applied_delay >= 0);

#ifdef _DEBUG
        cerr << "ipol=" << ipol << " ichan=" << ichan
             << " delay=" << applied_delay << endl;
#endif

        in_data += applied_delay * input_ndim;

        float* out_data = output->get_datptr (ichan, ipol);

        for (uint64_t idat=0; idat < output_nfloat; idat++)
          out_data[idat] = in_data[idat];

      }
    }
  }
  function -> mark (output);
}
