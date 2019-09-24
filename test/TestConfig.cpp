#include "TestConfig.hpp"

namespace util {
  TestConfig::TestConfig ()
  {
    toml_config_loaded = false;
  }


  void TestConfig::load_toml_config (bool force)
  {
    if (!toml_config_loaded || force)
    {
      const std::string current_file = __FILE__;
      const std::size_t found = current_file.find_last_of("/\\");
      const std::string current_dir = current_file.substr(0, found + 1);

      const std::string test_data_dir = util::get_test_env_var("DSPSR_TEST_DIR", current_dir);
      if (util::config::verbose) {
        std::cerr << "util::TestConfig::load_toml_config: test_data_dir="
          << test_data_dir << std::endl;
      }
      std::string test_data_file_path = test_data_dir + "/test_config.toml";

      if (util::config::verbose) {
        std::cerr << "util::TestConfig::load_toml_config: test_data_file_path="
          << test_data_file_path << std::endl;
      }
      toml_config = util::load_toml(test_data_file_path);
      toml_config_loaded = true;
    }
  }

  std::vector<float> TestConfig::get_thresh ()
  {

    if (! toml_config_loaded) {
      load_toml_config();
    }

    std::vector<float> tol(2);

    tol[0] = (float) toml_config.get<double>("atol");
    tol[1] = (float) toml_config.get<double>("rtol");

    return tol;
  }
}
