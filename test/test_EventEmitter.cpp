#include <functional>
#include <string>
#include <vector>

#include "catch.hpp"

#include "EventEmitter.h"

TEST_CASE ("EventEmitter class works with function types and object types", "[unit][no_file][EventEmitter]") {

  SECTION ("EventEmitter works with function types ") {

    typedef std::function<void(const std::string&)> LambdaType;

    EventEmitter<LambdaType> emitter ;

    std::vector<std::string> responses;
    responses.push_back("hello");

    auto lambda_auto = [] (const std::string& a) { };
    LambdaType lambda_ex = [] (const std::string& a) { };

    // auto lambda_auto = [&responses] (const std::string& a) { responses.push_back(a); };
    // LambdaType lambda_ex = [&responses] (const std::string& a) { responses.push_back(a); };

    emitter.on("event0", lambda_auto);
    emitter.on("event0", lambda_ex);
    // emitter.on("event0", [&responses] (const std::string& a) { responses.push_back(a); });
    emitter.on("event0", [] (const std::string& a) { });

    emitter.emit("event0", "hello");
    // REQUIRE(responses.size() == 3);
  }


  SECTION ("EventEmitter works with objects") {

    class Functor {
    public:
      void operator() (const std::string& a) {
        // std::cerr << "Functor::operator(): here" << std::endl;
        // responses.push_back(a);
        // std::cerr << "Functor::operator(): responses.size()=" << responses.size() << std::endl;
      }
      std::vector<std::string> responses;
    };

    EventEmitter<Functor> emitter;

    Functor lambda;

    emitter.on("event0", lambda);

    emitter.emit("event0", "hello");
  }

}
