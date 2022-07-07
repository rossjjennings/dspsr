/***************************************************************************
 *
 *   Copyright (C) 2008 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/PScrunch.h"

using namespace std;

dsp::PScrunch::PScrunch ()
  : Transformation<TimeSeries,TimeSeries> ("PScrunch", outofplace)
{
  output_npol = 0;
}

void dsp::PScrunch::set_output_state (Signal::State _state)
{
  switch (_state)
  {
    case Signal::Intensity:  // Square-law detected total power (1 pol)
    case Signal::PP_State:   // Just PP
    case Signal::QQ_State:   // Just QQ
      output_npol = 1;
      break;
    case Signal::PPQQ:       // Square-law detected, two polarizations
      output_npol = 2;
      break;
    case Signal::Coherence:  // PP, QQ, Re[PQ], Im[PQ]
    case Signal::Stokes:     // Stokes I,Q,U,V
      throw Error (InvalidParam, "dsp::PScrunch::set_output_state",
                   "unsensible output state=%s", Signal::state_string (_state));
      break;
    default:
      throw Error (InvalidParam, "dsp::PScrunch::set_output_state",
                   "invalid output state=%s", Signal::state_string (_state));
  }

  state = _state;

  if (verbose)
    cerr << "dsp::PScrunch::set_output_state to "
         << Signal::state_string(state) << endl;
}

void dsp::PScrunch::set_engine( Engine* _engine )
{
  engine = _engine;
}

void dsp::PScrunch::prepare()
{
  if (engine)
  {
    engine->setup ();
  }

  if (output_npol == 0)
    throw Error (InvalidState, "dsp::PScrunch::prepare",
                 "output state has not been configured");

  if (input == output)
    throw Error (InvalidState, "dsp::PScrunch::prepare",
                 "only out-of-place is allowed");

  if (input->get_ndim() != 1)
    throw Error (InvalidState, "dsp::PScrunch::prepare",
                 "invalid input ndim=%d", input->get_ndim());

  if (input->get_npol() != 2 && input->get_npol() != 4)
    throw Error (InvalidState, "dsp::PScrunch::prepare",
		 "invalid npol=%d", input->get_npol());

  if (input->get_state() == Signal::Stokes)
    throw Error (InvalidState, "dsp::PScrunch::prepare",
		 "input state of Signal::Stokes not supported");
}

void dsp::PScrunch::prepare_output ()
{
  if (verbose)
    cerr << "dsp::PScrunch::prepare_output()" << endl;

  if (input == output)
    throw Error (InvalidState, "dsp::PScrunch::prepare_output",
                 "only out-of-place is allowed");

  output->copy_configuration(input);
  output->set_state(state);
  output->set_npol(output_npol);
  output->set_order(input->get_order());
  output->resize(input->get_ndat());
  output->set_zeroed_data(input->get_zeroed_data());
  output->set_input_sample(input->get_input_sample());
}

/*!
  \pre input TimeSeries must contain detected data
*/
void dsp::PScrunch::transformation ()
{
  if (verbose)
    cerr << "dsp::PScrunch::transformation" << endl;

  prepare_output();

  const uint64_t input_ndat  = input->get_ndat();
  const unsigned input_npol  = input->get_npol();
  const unsigned input_nchan = input->get_nchan();
  const uint64_t output_ndat = input_ndat;

  if (output_ndat)
  {
    switch (input->get_order())
    {
    case TimeSeries::OrderFPT:
    {
      if (verbose)
        cerr << "dsp::PScrunch::transformation input_order=TimeSeries::OrderFPT" << endl;
      if (engine)
        engine->fpt_pscrunch (get_input(), get_output());
      else
      {
        for (unsigned ichan=0; ichan < input_nchan; ichan++)
        {
	  const float* in_p0 = input->get_datptr (ichan, 0);
	  const float* in_p1 = input->get_datptr (ichan, 1);
	  float* out_p0 = output->get_datptr (ichan, 0);
	
          if (output_npol == 1)
          {
            if (state == Signal::Intensity)
            {
	      for (uint64_t idat=0; idat < output_ndat; idat++)
	        out_p0[idat] = in_p0[idat] + in_p1[idat];
            }
            else if (state == Signal::PP_State)
            {
	      for (uint64_t idat=0; idat < output_ndat; idat++)
	        out_p0[idat] = in_p0[idat];
            }
            else if (state == Signal::QQ_State)
            {
	      for (uint64_t idat=0; idat < output_ndat; idat++)
	        out_p0[idat] = in_p1[idat];
            }
          }
          else if (output_npol == 2)
          {
            float* out_p1 = output->get_datptr (ichan, 1);
            for (uint64_t idat=0; idat < output_ndat; idat++)
            {
              out_p0[idat] = in_p0[idat];
              out_p1[idat] = in_p1[idat];
            }
          }
        }
      }
      break;
    }
    case TimeSeries::OrderTFP:
    {
      if (verbose)
        cerr << "dsp::PScrunch::transformation input_order=TimeSeries::OrderTFP" << endl;

      if (engine)
        engine->tfp_pscrunch (get_input(), get_output());
      else
      {
        uint64_t i = 0;
        uint64_t o = 0;
        float* out_data = output->get_dattfp();
        const float* in_data = input->get_dattfp();
        for (uint64_t idat=0; idat < output_ndat; idat++)
        {
          for (unsigned ichan=0; ichan < input_nchan; ichan++)
          {
            const float p0 = in_data[i];
            const float p1 = in_data[i+1];
            if (output_npol == 1)
            {
              if (state == Signal::Intensity)
                out_data[o] = p0 + p1;
              else if (state == Signal::PP_State)
                out_data[o] = p0;
              else if (state == Signal::QQ_State)
                out_data[o] = p1;
            }
            else if (output_npol == 2)
            {
              out_data[o] = p0;
              out_data[o+1] = p1;
            }
            i += input_npol;
            o += output_npol;
          }
        }
      }
      break;
    }
    default:
      throw Error (InvalidState, "dsp::PScrunch::operate",
		   "Can only handle data ordered TFP or FPT");
    }
  }
}

