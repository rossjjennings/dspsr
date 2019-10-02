#ifndef __TestReporter_hpp
#define __TestReporter_hpp

#include "util/util.hpp"

void check_error (const char*);

namespace test {
namespace util {

  template<class ReporterBase, class T>
  class TestReporter : public ReporterBase {

  public:

    typedef T DataType;

    TestReporter (bool _iscuda = false): iscuda(_iscuda) {}

    TestReporter (cudaStream_t _stream): stream(_stream) { iscuda = true; }

    void operator() (
      T* arr,
      unsigned nchan, unsigned npol, unsigned ndat, unsigned ndim)
    {
      unsigned total_size = nchan * npol * ndat * ndim;
      if (test::util::config::verbose)
      {
        std::cerr << "TestReporter::operator() (..."
          // << arr << ", "
          << nchan << ", "
          << npol << ", "
          << ndat << ", "
          << ndim << ")"
          << std::endl;
        std::cerr << "TestReporter::operator() total_size=" << total_size << std::endl;
      }
      std::vector<T> data (total_size);
      if (iscuda) {
        T* data_ptr = data.data();
        size_t total_size_bytes = total_size * sizeof(T);
        cudaError error;
        if (stream) {
          error = cudaMemcpyAsync(data_ptr, arr, total_size_bytes, cudaMemcpyDeviceToHost, stream);
          if (error != 0) {
            throw "TestReporter::operator() cudaMemcpyAsync error";
          }
          error = cudaStreamSynchronize(stream);
        } else {
          error = cudaMemcpy(data_ptr, arr, total_size_bytes, cudaMemcpyDeviceToHost);
          if (error != 0) {
            throw "TestReporter::operator() cudaMemcpy error";
          }
          error = cudaThreadSynchronize();
        }
        check_error("TestReporter::operator()");
      } else {
        data.assign(arr, arr + total_size);
      }
      data_vectors.push_back(data);
    }


    void concatenate_data_vectors (std::vector<T>& result)
    {
      unsigned n_vec = data_vectors.size();
      unsigned size_each = data_vectors[0].size();
      unsigned total_size = size_each * n_vec;
      result.resize(total_size);

      for (unsigned ipart=0; ipart<n_vec; ipart++) {
        for (unsigned idat=0; idat<size_each; idat++) {
          result[ipart * size_each + idat] = data_vectors[ipart][idat];
        }
      }

    }

    cudaStream_t stream;
    bool iscuda;
    std::vector<std::vector<T>> data_vectors;

  };

}
}

#endif
