#ifndef __TestConfig_hpp
#define __TestConfig_hpp

#include <vector>

#include "util/util.hpp"

namespace test {
namespace util {

  class TestConfig {

  public:

    TestConfig ();

    std::vector<float> get_thresh ();

    void load_toml_config (bool force=false);

    template<class T>
    T get_field (const std::string& field_name) {
      if (! toml_config_loaded) {
        load_toml_config();
      }

      return (T) toml_config.get<typename util::tomltype<T>::type>(field_name);
    }

  protected:

    toml::Value toml_config;

    bool toml_config_loaded;

  };
}
}

#endif
