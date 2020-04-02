#ifndef __TransformationProxy_hpp
#define __TransformationProxy_hpp

#include "dsp/TimeSeries.h"

namespace test {
namespace util {
  class TransformationProxy  {

  public:

    TransformationProxy ();

    TransformationProxy (
      unsigned _input_nchan,
      unsigned _output_nchan,
      unsigned _input_npol,
      unsigned _output_npol,
      unsigned _input_ndim,
      unsigned _output_ndim,
      unsigned _input_ndat,
      unsigned _output_ndat
    );

    virtual void setup (dsp::TimeSeries* _input, dsp::TimeSeries* _output);

    dsp::TimeSeries* get_input () const { return input; }

    dsp::TimeSeries* get_output () const { return output; }

    unsigned input_nchan;
    unsigned output_nchan;

    unsigned input_npol;
    unsigned output_npol;

    unsigned input_ndim;
    unsigned output_ndim;

    unsigned input_ndat;
    unsigned output_ndat;

  protected:

    dsp::TimeSeries* input;
    dsp::TimeSeries* output;
  };
}
}


#endif
