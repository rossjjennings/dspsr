//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2019 by Dean Shaff, Andrew Jameson and Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __EventEmitter_h
#define __EventEmitter_h

#include <string>
#include <vector>
#include <map>
#include <iostream>


template<typename FuncType>
class EventEmitter {

public:

  EventEmitter ();

  void on (const std::string& event_name, FuncType func);

  void on (const std::string& event_name, FuncType* func);

  template<class ... Types>
  void emit (const std::string& event_name, Types ... args);

private:

  std::map<std::string, std::vector<FuncType*>> event_map;

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

#endif
