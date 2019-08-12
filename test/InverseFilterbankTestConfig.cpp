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
  }
}
