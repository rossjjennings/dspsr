#include "InverseFilterbankTestConfig.hpp"

namespace util {
  namespace InverseFilterbank {

    std::vector<util::TestShape> InverseFilterbankTestConfig::get_test_vector_shapes ()
    {
      if (! toml_config_loaded) {
        load_toml_config();
      }

      const toml::Array toml_shapes = toml_config.get<toml::Array>("InverseFilterbank.test_shapes");

      std::vector<util::TestShape> vec(toml_shapes.size());
      util::TestShape sh;

      for (unsigned idx=0; idx < toml_shapes.size(); idx++)
      {
        util::from_toml(toml_shapes[idx], sh);
        vec[idx] = sh;
      }

      if (util::config::verbose) {
        std::cerr << "InverseFilterbankTestConfig::get_test_vector_shapes: vec.size()=" << vec.size() << std::endl;
      }

      return vec;
    }

    void InverseFilterbankProxy::setup (
      dsp::TimeSeries* in,
      dsp::TimeSeries* out,
      bool do_fft_window,
      bool do_response
    )
    {
      if (util::config::verbose) {
        std::cerr << "util::InverseFilterbankProxy::setup: input_ndat=" << input_ndat << std::endl;
        std::cerr << "util::InverseFilterbankProxy::setup: input_overlap=" << input_overlap << std::endl;
      }

      auto os_in2out = [this] (unsigned n) -> unsigned {
        return this->os_factor.normalize(n) * this->input_nchan / this->output_nchan;
      };
      unsigned input_fft_length = input_ndat;
      unsigned output_fft_length = os_in2out(input_fft_length);
      unsigned output_overlap = os_in2out(input_overlap);

      std::vector<unsigned> in_dim = {
        input_nchan, npol, input_fft_length*npart};
      std::vector<unsigned> out_dim = {
        output_nchan, npol, output_fft_length*npart};

      unsigned in_size = util::product(in_dim);
      unsigned out_size = util::product(out_dim);

      std::vector<std::complex<float>> in_vec(in_size);
      std::vector<std::complex<float>> out_vec(out_size);

      auto random_gen = util::random<float>();

      for (unsigned idx=0; idx<in_size; idx++) {
        // in_vec[idx] = std::complex<float>((float) idx, (float) idx);
        in_vec[idx] = std::complex<float>(random_gen(), random_gen());
        // std::cerr << in_vec[idx] << std::endl;
      }

      util::loadTimeSeries<std::complex<float>>(in_vec, in, in_dim);
      util::loadTimeSeries<std::complex<float>>(out_vec, out, out_dim);

      if (do_fft_window)
      {
        if (util::config::verbose) {
          std::cerr << "util::InverseFilterbankProxy::setup creating Tukey FFT window" << std::endl;
        }
        Reference::To<dsp::Apodization> fft_window = new dsp::Apodization;
        fft_window->Tukey(input_fft_length, 0, input_overlap, true);
        filterbank->set_apodization(fft_window);
      }

      if (do_response)
      {
        if (util::config::verbose) {
          std::cerr << "util::InverseFilterbankProxy::setup creating Response" << std::endl;
        }
        Reference::To<dsp::Response> response = new dsp::Response;
        response->resize(1, output_nchan, output_fft_length, 2);
        float* ptr;

        for (unsigned ichan=0; ichan<response->get_nchan(); ichan++) {
          ptr = response->get_datptr(ichan, 0);
          for (unsigned idat=0; idat<response->get_ndat(); idat++) {
            ptr[2*idat] = random_gen();
            ptr[2*idat + 1] = random_gen();
          }
        }
        filterbank->set_response(response);
      }


      filterbank->set_input(in);
    	filterbank->set_output(out);

      filterbank->set_oversampling_factor(os_factor);
      filterbank->set_input_fft_length(input_fft_length);
      filterbank->set_output_fft_length(output_fft_length);
      filterbank->set_input_discard_pos(input_overlap);
      filterbank->set_input_discard_neg(input_overlap);
      filterbank->set_output_discard_pos(output_overlap);
      filterbank->set_output_discard_neg(output_overlap);
    }

    float* InverseFilterbankProxy::allocate_scratch (
      unsigned total_scratch_needed
    )
    {
      float* space = scratch->space<float> (total_scratch_needed);
      return space;
    }

  }
}
