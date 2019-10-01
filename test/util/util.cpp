#include <unistd.h> // for getcwd
#include <cstdlib>
#include <string>

#include "dsp/Shape.h"

#include "util/util.hpp"

void check_error (const char*);

bool test::util::config::verbose = 0;

std::chrono::time_point<std::chrono::high_resolution_clock> test::util::now ()
{
  return std::chrono::high_resolution_clock::now();
}

void test::util::load_psr_data (
  dsp::IOManager manager,
  unsigned block_size,
  dsp::TimeSeries* ts,
  int nblocks)
{
  dsp::Input* input = manager.get_input();
  input->set_block_size(block_size);
  if (nblocks == -1) {
    while (! manager.get_input()->eod()) {
      manager.load(ts);
    }
  } else {
    for (int iblock=0; iblock<nblocks; iblock++) {
      manager.load(ts);
    }
  }
}

void test::util::set_verbose (bool val)
{
  dsp::Input::verbose = val;
  dsp::Operation::verbose = val;
  dsp::Observation::verbose = val;
  dsp::Shape::verbose = val;
  test::util::config::verbose = val;
}

std::string test::util::get_test_env_var (const std::string& env_var_name, const std::string& default_val)
{
  const char* env_var = std::getenv(env_var_name.c_str());
  if (env_var) {
    return std::string(env_var);
  } else {
    return default_val;
  }
}

std::string test::util::get_test_data_dir ()
{
  return test::util::get_test_env_var("DSPSR_TEST_DATA_DIR");
}

std::string test::util::get_working_path ()
{
  char temp[FILENAME_MAX];
  return (getcwd(temp, sizeof(temp)) ? std::string( temp ) : std::string(""));
}


std::function<void(dsp::TimeSeries*, dsp::TimeSeries*, cudaMemcpyKind)> test::util::transferTimeSeries (
  cudaStream_t stream, CUDA::DeviceMemory* memory
)
{
  return [stream, memory] (dsp::TimeSeries* in, dsp::TimeSeries* out, cudaMemcpyKind k){
    if (test::util::config::verbose)
    {
      std::cerr << "test::util::transferTimeSeries lambda(" << in << ", " << out << ", " << k << ")" << std::endl;
    }
    dsp::TransferCUDA transfer(stream);
    transfer.set_kind(k);
    if (k == cudaMemcpyHostToDevice) {
      if (test::util::config::verbose) {
        std::cerr << "test::util::transferTimeSeries setting output memory" << std::endl;
      }
      out->set_memory(memory);
    }
    transfer.set_input(in);
    transfer.set_output(out);
    if (test::util::config::verbose) {
      std::cerr << "test::util::transferTimeSeries prepare" << std::endl;
    }
    transfer.prepare();
    if (test::util::config::verbose) {
      std::cerr << "test::util::transferTimeSeries operate" << std::endl;
    }
    transfer.operate();
    check_error("test::util::transferTimeSeries lambda");
  };
}

bool test::util::allclose (dsp::TimeSeries* a, dsp::TimeSeries* b, float atol, float rtol)
{
  if (test::util::config::verbose) {
    std::cerr << "test::util::allclose(dsp::TimeSeries*, dsp::TimeSeries*)" << std::endl;
  }

  bool allclose = true;

  std::vector<std::string> shape_str = {"nchan", "npol", "ndat", "ndim"};
  std::vector<unsigned> a_shape = {
      a->get_nchan(), a->get_npol(), (unsigned) a->get_ndat(), a->get_ndim()
  };

  std::vector<unsigned> b_shape = {
      b->get_nchan(), b->get_npol(), (unsigned) b->get_ndat(), b->get_ndim()
  };
  if (test::util::config::verbose) {

    auto print_shape = [] (std::vector<unsigned> shape, std::string name) {
      std::cerr << "test::util::allclose:" << name << " shape=(";
      for (unsigned idx=0; idx<shape.size(); idx++) {
        std::cerr << shape[idx];
        if (idx == shape.size() - 1) {
          std::cerr << ")";
        } else {
          std::cerr << " ";
        }
      }
      std::cerr << std::endl;
    };

    print_shape (a_shape, "a");
    print_shape (b_shape, "b");
  }



  for (unsigned idx=0; idx<shape_str.size(); idx++) {
    if (a_shape[idx] != b_shape[idx])
    {
      std::cerr << "test::util::allclose: " << shape_str[idx] << " dim not equal: "<< a_shape[idx] << " != " << b_shape[idx] << std::endl;
      return false;
    }
  }

  std::complex<float>* a_ptr;
  std::complex<float>* b_ptr;

  for (unsigned ichan=0; ichan<a->get_nchan(); ichan++) {
    for (unsigned ipol=0; ipol<a->get_npol(); ipol++) {
      // std::cerr << ichan << ", " << ipol << std::endl;
      a_ptr = reinterpret_cast<std::complex<float>*> (a->get_datptr(ichan, ipol));
      b_ptr = reinterpret_cast<std::complex<float>*> (b->get_datptr(ichan, ipol));
      for (unsigned idat=0; idat<a->get_ndat(); idat++) {
        // std::cerr << "[" << *a_ptr << ", " << *b_ptr << "] ";
        if (! test::util::isclose(*a_ptr, *b_ptr, atol, rtol)) {
          // if (test::util::config::verbose) {
          //   std::cerr << "[(" << ichan << ", " << ipol << ", " << idat << ")="
          //     << *a_ptr << ", " << *b_ptr << ", " << abs(*a_ptr - *b_ptr) << ", "
          //     << abs(*a_ptr - *b_ptr) / abs(*b_ptr) << "]" << std::endl;
          // }
          allclose = false;
          // break;
        }
        a_ptr++;
        b_ptr++;
      }
      // if (test::util::config::verbose)
      // {
        // std::cerr << std::endl;
      // }
    }
  }
  return allclose;
}

// void test::util::to_json(json& j, const test::util::TestShape& sh) {
//   j = json{
//       {"npart", sh.npart},
//       {"input_nchan", sh.input_nchan},
//       {"output_nchan", sh.output_nchan},
//       {"input_npol", sh.input_npol},
//       {"output_npol", sh.output_npol},
//       {"input_ndat", sh.input_ndat},
//       {"input_ndat", sh.input_ndat},
//       {"overlap_pos", sh.overlap_pos},
//       {"overlap_neg", sh.overlap_neg}
//   };
// }
//
// void test::util::from_json(const json& j, test::util::TestShape& sh) {
//   sh.npart = j["npart"].get<unsigned>();
//   sh.input_nchan = j["input_nchan"].get<unsigned>();
//   sh.output_nchan = j["output_nchan"].get<unsigned>();
//   sh.input_npol = j["input_npol"].get<unsigned>();
//   sh.output_npol = j["output_npol"].get<unsigned>();
//   sh.input_ndat = j["input_ndat"].get<unsigned>();
//   sh.output_ndat = j["output_ndat"].get<unsigned>();
//   sh.overlap_pos = j["overlap_pos"].get<unsigned>();
//   sh.overlap_neg = j["overlap_neg"].get<unsigned>();
//   // j.at("npart").get_to(sh.npart);
//   // j.at("input_nchan").get_to(sh.input_nchan);
//   // j.at("output_nchan").get_to(sh.output_nchan);
//   // j.at("input_npol").get_to(sh.input_npol);
//   // j.at("output_npol").get_to(sh.output_npol);
//   // j.at("input_ndat").get_to(sh.input_ndat);
//   // j.at("output_ndat").get_to(sh.output_ndat);
//   // j.at("overlap_pos").get_to(sh.overlap_pos);
//   // j.at("overlap_neg").get_to(sh.overlap_neg);
// }
//
// json test::util::load_json (std::string file_path)
// {
//
//   std::ifstream in_stream(file_path);
//   if (! in_stream.good()) {
//     throw "test::util::load_json: file_path is either nonexistent or locked";
//   }
//   json j;
//   in_stream >> j;
//   in_stream.close();
//   return j;
// }

const toml::Value test::util::load_toml (const std::string& file_path)
{
  std::ifstream ifs(file_path);
  if (! ifs.good()) {
    throw "test::util::load_toml: file_path is either nonexistent or locked";
  }
  toml::ParseResult pr = toml::parse(ifs);
  ifs.close();

  if (! pr.valid())
  {
    throw "test::util::load_toml: invalid TOML file";
  }
  const toml::Value result = pr.value;
  return result;
}


void test::util::from_toml (const toml::Value& val, test::util::TestShape& sh)
{
  sh.npart = (unsigned) val.get<int>("npart");
  sh.input_nchan = (unsigned) val.get<int>("input_nchan");
  sh.output_nchan = (unsigned) val.get<int>("output_nchan");
  sh.input_npol = (unsigned) val.get<int>("input_npol");
  sh.output_npol = (unsigned) val.get<int>("output_npol");
  sh.input_ndat = (unsigned) val.get<int>("input_ndat");
  sh.output_ndat = (unsigned) val.get<int>("output_ndat");
  sh.overlap_pos = (unsigned) val.get<int>("overlap_pos");
  sh.overlap_neg = (unsigned) val.get<int>("overlap_neg");
}

void test::util::to_toml (toml::Value& val, const test::util::TestShape& sh)
{
  val.set("npart", (int) sh.npart);
  val.set("input_nchan", (int) sh.input_nchan);
  val.set("output_nchan", (int) sh.output_nchan);
  val.set("input_npol", (int) sh.input_npol);
  val.set("output_npol", (int) sh.output_npol);
  val.set("input_ndat", (int) sh.input_ndat);
  val.set("output_ndat", (int) sh.output_ndat);
  val.set("overlap_pos", (int) sh.overlap_pos);
  val.set("overlap_neg", (int) sh.overlap_neg);
}
