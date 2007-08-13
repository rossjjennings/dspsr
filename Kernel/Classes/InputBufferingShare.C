/***************************************************************************
 *
 *   Copyright (C) 2007 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
#include "dsp/InputBufferingShare.h"
#include "ThreadContext.h"

using namespace std;

dsp::InputBuffering::Share::Share ()
{
  name = "InputBuffering::Share";
  context = 0;
  context_owner = false;
}

dsp::InputBuffering::Share::Share (InputBuffering* _buffer,
				   HasInput<TimeSeries>* _target)
{
  buffer = _buffer;
  target = _target;

  context = new ThreadContext;
  context_owner = true;

  name = "InputBuffering::Share";
}

dsp::InputBuffering::Share*
dsp::InputBuffering::Share::clone (HasInput<TimeSeries>* _target)
{
  Share* result = new Share;
  result -> buffer = buffer;
  result -> target = _target;
  result -> context = context;

  return result;
}

dsp::InputBuffering::Share::~Share ()
{
  if (context && context_owner)
    delete context;
}

//! Set the minimum number of samples that can be processed
void dsp::InputBuffering::Share::set_minimum_samples (uint64 samples)
try {
  ThreadContext::Lock lock (context);

  buffer->set_target(target);
  buffer->set_minimum_samples (samples);
}
 catch (Error& error) {
   throw error += "dsp::InputBuffering::Share::set_minimum_samples";
 }

/*! Copy remaining data from the target Transformation's input to buffer */
void dsp::InputBuffering::Share::set_next_start (uint64 next)
try {
  ThreadContext::Lock lock (context);

  buffer->set_target (target);
  buffer->set_next_start (next);

  if (context)
    context->broadcast();
}
 catch (Error& error) {
   throw error += "dsp::InputBuffering::Share::set_next_start";
 }

/*! Prepend buffered data to target Transformation's input TimeSeries */
void dsp::InputBuffering::Share::pre_transformation ()
try {

  ThreadContext::Lock lock (context);

  int64 want = target->get_input()->get_input_sample();

  // don't wait for data preceding the first loaded block
  if (want == 0)
    return;

  while ( buffer->get_next_contiguous() != want ) {

    if (Operation::verbose)
      cerr << "dsp::InputBuffering::Share::pre_transformation want=" << want 
	   << "; have=" << buffer->get_next_contiguous() << endl;

    context->wait();

  }

  buffer->set_target (target);
  buffer->pre_transformation ();
}
 catch (Error& error) {
   throw error += "dsp::InputBuffering::Share::pre_transformation";
 }

/*! No action required after transformation */
void dsp::InputBuffering::Share::post_transformation ()
{
}

