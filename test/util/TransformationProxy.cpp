#include "util.hpp"

#include "util/TransformationProxy.hpp"

test::util::TransformationProxy::TransformationProxy () {}

test::util::TransformationProxy::TransformationProxy (
  unsigned _input_nchan,
  unsigned _output_nchan,
  unsigned _input_npol,
  unsigned _output_npol,
  unsigned _input_ndim,
  unsigned _output_ndim,
  unsigned _input_ndat,
  unsigned _output_ndat
): input_nchan(_input_nchan),
  output_nchan(_output_nchan),
  input_npol(_input_npol),
  output_npol(_output_npol),
  input_ndim(_input_ndim),
  output_ndim(_output_ndim),
  input_ndat(_input_ndat),
  output_ndat(_output_ndat) {}

void test::util::TransformationProxy::setup (
  dsp::TimeSeries* _input,
  dsp::TimeSeries* _output
)
{
  _input->set_state (Signal::Analytic);
  _input->set_nchan (input_nchan);
  _input->set_npol (input_npol);
  _input->set_ndat (input_ndat);
  _input->set_ndim (input_ndim);
  _input->resize (input_ndat);

  _output->set_state (Signal::Analytic);
  _output->set_nchan (output_nchan);
  _output->set_npol (output_npol);
  _output->set_ndat (output_ndat);
  _output->set_ndim (output_ndim);
  _output->resize (output_ndat);

  auto random_gen = test::util::random<float>();
  float* in_ptr;

  for (unsigned ichan = 0; ichan < _input->get_nchan(); ichan++) {
    for (unsigned ipol = 0; ipol < _input->get_npol(); ipol++) {
      in_ptr = _input->get_datptr(ichan, ipol);
      for (unsigned idat = 0; idat < _input->get_ndat(); idat++) {
        in_ptr[2*idat] = random_gen();
        in_ptr[2*idat + 1] = random_gen();
      }
    }
  }

  input = _input;
  output = _output;

}
