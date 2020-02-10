//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2019 by Dean Shaff, Andrew Jameson and Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __EventEmitter_h
#define __EventEmitter_h

#if HAVE_CONFIG_H
#include <config.h>
#endif

#include <string>
#include <vector>
#include <map>
#include <iostream>


#if HAVE_CXX11
//! Implements a simple event emitter class. This is C++11 only.
//!
//! Example Usage:
//!
//!   EventEmitter<std::function<void(int)> emitter;
//!
//!   auto lam_auto = [] (int a) { std::cerr << a << std::endl; };
//!   std::function<void(int)> lam = [] (int a) { std::cerr << a << std::endl; };
//!   emitter.on("event0", lam);
//!   emitter.on("event0", lam_auto);
//!   emitter.on("event0", [] (int a) { std::cerr << a << std::endl; });
//!
//!   emitter.emit("event0");
//!
template<typename FuncType>
class EventEmitter {

public:

  EventEmitter ();

  void on (const std::string& event_name, FuncType func);

  void on (const std::string& event_name, FuncType* func);

  template<class ... Types>
  void emit (const std::string& event_name, Types ... args);

private:

  std::map<std::string, std::vector<FuncType*> > event_map;

};


template<typename FuncType>
EventEmitter<FuncType>::EventEmitter ()
{ }

template<typename FuncType>
void EventEmitter<FuncType>::on(const std::string& event_name, FuncType _func)
{
  on(event_name, &_func);
}

template<typename FuncType>
void EventEmitter<FuncType>::on(const std::string& event_name, FuncType* _func)
{
  if (event_map.find(event_name) == event_map.end()) {
    event_map[event_name] = {_func};
  } else {
    event_map[event_name].push_back(_func);
  }
}

template<typename FuncType>
template<class ... Types>
void EventEmitter<FuncType>::emit (const std::string& event_name, Types ... args)
{
  if (event_map.find(event_name) != event_map.end())
  {
    auto func_b = event_map[event_name].begin();
    auto func_e = event_map[event_name].end();
    for (; func_b != func_e; func_b++) {
      (**func_b)(args...);
    }
  }
}

#else


//! pre C++11 mock implementation of EventEmitter class.
//!
//! Example Usage:
//!
//!   EventEmitter<std::function<void(int)> emitter;
//!
//!   auto lam_auto = [] (int a) { std::cerr << a << std::endl; };
//!   std::function<void(int)> lam = [] (int a) { std::cerr << a << std::endl; };
//!   emitter.on("event0", lam);
//!   emitter.on("event0", lam_auto);
//!   emitter.on("event0", [] (int a) { std::cerr << a << std::endl; });
//!
//!   emitter.emit("event0");
//!
template<typename FuncType>
class EventEmitter {

public:

  EventEmitter ();

  void on (const std::string& event_name, FuncType func);

  void on (const std::string& event_name, FuncType* func);

  void emit (const std::string& event_name, ...);

private:

  std::map<std::string, std::vector<FuncType*> > event_map;

};


template<typename FuncType>
EventEmitter<FuncType>::EventEmitter ()
{ }


template<typename FuncType>
void EventEmitter<FuncType>::on(const std::string& event_name, FuncType _func)
{
}

template<typename FuncType>
void EventEmitter<FuncType>::on(const std::string& event_name, FuncType* _func)
{
}

template<typename FuncType>
void EventEmitter<FuncType>::emit (const std::string& event_name, ...)
{
}
#endif

#endif
