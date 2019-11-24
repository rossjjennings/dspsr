#include <complex>
#include <vector>
#include <iostream>

#include "dsp/InverseFilterbankEngineCPU.h"
#include "dsp/InverseFilterbankEngineCUDA.h"
#include "dsp/InverseFilterbank.h"
#include "dsp/TransferCUDA.h"
#include "dsp/MemoryCUDA.h"
#include "Rational.h"
#include "CommandLine.h"

#include "util/util.hpp"
#include "InverseFilterbankTestConfig.hpp"

class BenchmarkOptions : public Reference::Able {
public:
  unsigned niter = 10;
  bool cuda_only = false;
  bool cpu_only = false;
  bool report = false;
  bool output_toml = false;
  // npart, input_nchan, output_nchan, input_npol, output_npol, input_ndat, output_ndat, overlap_pos, overlap_neg
  test::util::TestShape test_shape{10, 256, 1, 2, 2, 128, 0, 16, 16};

  void set_verbose () {
    test::util::config::verbose = true;
  }

  void set_very_verbose () {
    set_verbose();
    test::util::set_verbose(true);
  }

  void set_cuda_only () {
    cuda_only = true;
  }

  void set_cpu_only () {
    cpu_only = true;
  }

  void set_report () {
    report = true;
  }

  void set_output_toml () {
    output_toml = true;
  }

};

void check_error (const char*);

void parse_options (BenchmarkOptions* options, int argc, char** argv)
{
  CommandLine::Menu menu;
  CommandLine::Argument* arg;

  menu.set_help_header ("benchmark InverseFilterbank CUDA and CPU engines");

  arg = menu.add (options->niter, "n", "niter");
  arg->set_help ("Specify the number of iterations to run [default:10]");

  arg = menu.add (options->test_shape.input_npol, "npol");
  arg->set_help ("Specify the number of input polarisations [default:2]");

  arg = menu.add (options->test_shape.npart, "npart");
  arg->set_help ("Specify the number of parts [default:10]");

  arg = menu.add (options->test_shape.input_nchan, "input_nchan");
  arg->set_help ("Specify the number of input channels [default:256]");

  arg = menu.add (options->test_shape.output_nchan, "output_nchan");
  arg->set_help ("Specify the number of output channels [default:1]");

  arg = menu.add (options->test_shape.input_ndat, "input_ndat");
  arg->set_help ("Specify the size of the forward FFT [default:128]");

  arg = menu.add (options->test_shape.overlap_pos, "overlap");
  arg->set_help ("Specify the size of the input overlap region [default:16]");

  arg = menu.add (options, &BenchmarkOptions::set_verbose, "v");
  arg->set_help ("verbose mode");

  arg = menu.add (options, &BenchmarkOptions::set_very_verbose, "V");
  arg->set_help ("very verbose mode");

  arg = menu.add (options, &BenchmarkOptions::set_cuda_only, "cuda-only");
  arg->set_help ("cuda only");

  arg = menu.add (options, &BenchmarkOptions::set_report, "r");
  arg->set_help ("run CUDA kernels in verbose mode");

  arg = menu.add (options, &BenchmarkOptions::set_output_toml, "toml");
  arg->set_help ("output TOML report");

  menu.parse (argc, argv);

  if (options->test_shape.overlap_pos > options->test_shape.input_ndat/2)
  {
    throw "Overlap region cannot be larger than input FFT";
  }
  if (options->test_shape.output_nchan >= options->test_shape.input_nchan)
  {
    throw "Number of output channels cannot be greater than input channels";
  }

}

int main (int argc, char** argv)
{
  test::util::time_point t0;
  test::util::time_point t1;

  // dsp::Operation::record_time = true;

  Reference::To<BenchmarkOptions> options = new BenchmarkOptions;

  parse_options (options, argc, argv);
  void* stream = 0;
  cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  CUDA::DeviceMemory* device_memory = new CUDA::DeviceMemory(cuda_stream);

  CUDA::InverseFilterbankEngineCUDA engine_cuda(cuda_stream);
  dsp::InverseFilterbankEngineCPU engine_cpu;

  engine_cuda.set_record_time(true);
  engine_cuda.set_report(options->report);
  engine_cpu.set_report(options->report);

  Reference::To<dsp::TimeSeries> in = new dsp::TimeSeries;
  Reference::To<dsp::TimeSeries> out = new dsp::TimeSeries;
  Reference::To<dsp::TimeSeries> in_gpu = new dsp::TimeSeries;
  Reference::To<dsp::TimeSeries> out_gpu = new dsp::TimeSeries;

  Rational os_factor (4, 3);
  unsigned npart = options->test_shape.npart;
  unsigned npol = options->test_shape.input_npol;

  uint64_t nsamples = 2 * options->test_shape.input_nchan *
                      options->test_shape.input_npol *
                      options->test_shape.input_ndat *
                      options->test_shape.npart;
  uint64_t nbytes = sizeof(float) * nsamples;
  float ngigabits = (float) (8 * nbytes) / std::giga::num;
  std::cerr << "Processing " << ngigabits << " Gbits of data" << std::endl;

  test::util::InverseFilterbank::InverseFilterbankProxy proxy(
    os_factor, npart, npol,
    options->test_shape.input_nchan, options->test_shape.output_nchan,
    options->test_shape.input_ndat, options->test_shape.overlap_pos
  );

  // config.filterbank->set_device(device_memory);
  proxy.filterbank->set_pfb_dc_chan(true);
  proxy.filterbank->set_pfb_all_chan(true);

  bool do_fft_window = true;
  bool do_response = true;
  t0 = test::util::now();
  proxy.setup(in, out, do_fft_window, do_response);
  test::util::delta<std::ratio<1>> delta_setup_initial = test::util::now() - t0;

  std::cerr << "initial setup took " << delta_setup_initial.count() << " s" << std::endl;


  // if (test::util::config::verbose) {
  //   std::cerr << "in shape=("
  //     << in->get_nchan() << ","
  //     << in->get_npol() << ","
  //     << in->get_ndat() << ","
  //     << in->get_ndim() << ")"
  //     << std::endl;
  // }



  test::util::delta<std::ratio<1>> delta_cpu;

  if (! options->cuda_only)
  {
    engine_cpu.setup(proxy.filterbank);
    float* scratch_cpu = proxy.allocate_scratch(engine_cpu.get_total_scratch_needed());
    engine_cpu.set_scratch(scratch_cpu);

    test::util::delta<std::ratio<1>> delta;
    t0 = test::util::now();
    for (unsigned iiter=0; iiter<options->niter; iiter++) {
      t1 = test::util::now();
      engine_cpu.perform(
        in, out, npart
      );
      engine_cpu.finish();
      delta = test::util::now() - t1;
      if (test::util::config::verbose) {
        std::cerr << "CPU engine loop " << iiter << " took " << delta.count() << "s" << std::endl;
      }
    }

    delta_cpu = test::util::now() - t0;
    float delta_cpu_s_per_loop = (float) delta_cpu.count() / options->niter;

    std::cerr << "CPU engine took " << delta_cpu_s_per_loop << " s per loop"
      << std::endl;
    std::cerr << "CPU engine processing @ " << ngigabits / delta_cpu_s_per_loop << " Gbits/s"
      << std::endl;

  }

  auto transfer = test::util::transferTimeSeries(cuda_stream, device_memory);
  t0 = test::util::now();
  transfer(in, in_gpu, cudaMemcpyHostToDevice);
  transfer(out, out_gpu, cudaMemcpyHostToDevice);
  test::util::delta<std::ratio<1>> delta_transfer = test::util::now() - t0;

  std::cerr << "transfering from CPU to GPU took " << delta_transfer.count() << " s" << std::endl;

  t0 = test::util::now();
  engine_cuda.setup(proxy.filterbank);
  proxy.set_memory<CUDA::DeviceMemory>(device_memory);
  float* scratch_cuda = proxy.allocate_scratch(engine_cuda.get_total_scratch_needed());
  engine_cuda.set_scratch(scratch_cuda);
  test::util::delta<std::ratio<1>> delta_setup = test::util::now() - t0;

  std::cerr << "setting up CUDA engine took " << delta_setup.count() << " s" << std::endl;

  // t0 = test::util::now();
  float delta_cuda_i = 0;
  float delta_cuda = 0;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);


  for (unsigned iiter=0; iiter<options->niter; iiter++) {
    // t1 = test::util::now();
    cudaEventRecord(start);
    engine_cuda.perform(
      in_gpu, out_gpu, npart
    );
    engine_cuda.finish();
    // test::util::delta<std::ratio<1>> delta = test::util::now() - t1;
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&delta_cuda_i, start, stop);
    delta_cuda += delta_cuda_i;
    // cudaDeviceSynchronize();
    // test::util::delta<std::ratio<1>> delta = test::util::now() - t0;
    if (test::util::config::verbose) {
      std::cerr << "CUDA engine loop " << iiter << " took " << delta_cuda_i/1000 << "s" << std::endl;
      // std::cerr << "CUDA engine (CPU timer) loop " << iiter << " took " << delta.count() << "s" << std::endl;
    }
  }

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  // test::util::delta<std::ratio<1>> delta_cuda = test::util::now() - t0;
  float delta_cuda_s = delta_cuda / 1000;
  float delta_cuda_s_per_loop = delta_cuda_s / options->niter;
  float rate_cuda = ngigabits / delta_cuda_s_per_loop;
  std::cerr << "CUDA engine took " << delta_cuda_s_per_loop << " s per loop"
    << std::endl;
  std::cerr << "CUDA engine processing @ " << rate_cuda << " Gbits/s"
    << std::endl;

  // std::cerr << "CUDA engine " << delta_cuda.count() / options->niter << " s per loop"
  //   << std::endl;
  // std::cerr << "CUDA engine processing " << ngigabits / delta_cuda.count() / options->niter << " Gbits/s"
  //   << std::endl;

  float delta_cpu_s = (float) delta_cpu.count();

  if (! options->cuda_only) {
    if (delta_cpu_s > delta_cuda_s) {
      std::cerr << "CUDA engine " << delta_cpu_s / delta_cuda_s
        << " times faster" << std::endl;
    } else {
      std::cerr << "CPU engine " << delta_cuda_s / delta_cpu_s
        << " times faster" << std::endl;
    }
  }

  if (options->output_toml) {
    toml::Value output_val ((toml::Table()));
    toml::Value* root = &output_val;
    auto output_array = root->setChild("benchmark", toml::Array());
    auto inner_val = output_array->push((toml::Table()));

    test::util::to_toml(*inner_val, options->test_shape);
    inner_val->setChild("nbytes", (int64_t) nbytes);

    inner_val->setChild("niter", (int) options->niter);
    inner_val->setChild("CUDA_total_time", (double) delta_cuda_s);
    if (! options->cuda_only) {
      inner_val->setChild("CPU_total_time", (double) delta_cpu_s);
    }
    std::cout << output_val << std::endl;
  }


}
