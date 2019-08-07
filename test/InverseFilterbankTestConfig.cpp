#include "InverseFilterbankTestConfig.hpp"

namespace util {
  namespace InverseFilterbank {

    InverseFilterbankTestConfig::InverseFilterbankTestConfig ()
    {
      // json_config_loaded = false;
      toml_config_loaded = false;
    }

    // void InverseFilterbankTestConfig::load_json_config (bool force)
    // {
    //   if (!json_config_loaded || force)
    //   {
    //     std::string test_data_dir = util::get_test_env_var("DSPSR_TEST_DIR", "./test");
    //     std::string test_data_file_path = test_data_dir + "/test_config.json";
    //
    //     if (util::config::verbose) {
    //       std::cerr << "util::InverseFilterbank::load_json_config: test_data_file_path=" << test_data_file_path << std::endl;
    //     }
    //     json_config = util::load_json(test_data_file_path);
    //     json_config_loaded = true;
    //   }
    // }

    void InverseFilterbankTestConfig::load_toml_config (bool force)
    {
      if (!toml_config_loaded || force)
      {
        std::string test_data_dir = util::get_test_env_var("DSPSR_TEST_DIR", "./test");
        std::string test_data_file_path = test_data_dir + "/test_config.toml";

        if (util::config::verbose) {
          std::cerr << "util::InverseFilterbank::load_toml_config: test_data_file_path=" << test_data_file_path << std::endl;
        }
        toml_config = util::load_toml(test_data_file_path);
        toml_config_loaded = true;
      }
    }

    std::vector<float> InverseFilterbankTestConfig::get_thresh ()
    {

      // if (! json_config_loaded) {
      //   load_json_config();
      // }
      // std::vector<float> tol(2);
      //
      // tol[0] = json_config["InverseFilterbank"]["atol"].get<float>();
      // tol[1] = json_config["InverseFilterbank"]["rtol"].get<float>();
      //
      // return tol;

      if (! toml_config_loaded) {
        load_toml_config();
      }

      std::vector<float> tol(2);

      tol[0] = (float) toml_config.get<double>("InverseFilterbank.atol");
      tol[1] = (float) toml_config.get<double>("InverseFilterbank.rtol");

      return tol;
    }

    std::vector<util::TestShape> InverseFilterbankTestConfig::get_test_vector_shapes ()
    {

      // if (! json_config_loaded) {
      //   load_json_config();
      // }
      //
      // auto test_shapes = json_config["InverseFilterbank"]["test_shapes"];
      //
      // if (util::config::verbose) {
      //   std::cerr << "InverseFilterbankTestConfig::get_test_vector_shapes: test_shapes.size()=" << test_shapes.size() << std::endl;
      // }
      //
      //
      // std::vector<util::TestShape> vec(test_shapes.size());
      //
      // for (unsigned idx=0; idx < test_shapes.size(); idx++)
      // {
      //   vec[idx] = test_shapes[idx].get<util::TestShape>();
      // }
      //
      // if (util::config::verbose) {
      //   std::cerr << "InverseFilterbankTestConfig::get_test_vector_shapes: vec.size()=" << vec.size() << std::endl;
      // }
      //
      // return vec;

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
