/***************************************************************************
 *
 *   Copyright (C) 2020 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/FoldManager.h"
#include "dsp/Fold.h"

using namespace std;

dsp::FoldManager::FoldManager () : Operation("FoldManager")
{
}

void dsp::FoldManager::manage (Fold* a_fold)
{
  fold.push_back (a_fold);
}

void dsp::FoldManager::operation()
{
  if (verbose)
    cerr << "dsp::FoldManager::operation this=" << (void*) this << endl;

  for (unsigned i=0; i<fold.size(); i++)
  {
    if (verbose)
      cerr << "dsp::FoldManager::operation fold[" << i << "]=" 
           << (void*) fold[i].get() << endl;

    fold[i]->operate();
  }
}

void dsp::FoldManager::prepare ()
{
  for (unsigned i=0; i<fold.size(); i++)
    fold[i]->prepare();
}

void dsp::FoldManager::prepare (const Observation* observation)
{
  for (unsigned i=0; i<fold.size(); i++)
    fold[i]->prepare(observation);
}

void dsp::FoldManager::combine (const Operation* other)
{
  if (verbose)
    cerr << "dsp::FoldManager::combine this=" << (void*) this
         << " other=" << (void*) other << endl;

  const FoldManager* manager = dynamic_cast<const FoldManager*> (other);
  if (!manager)
  {
    if (verbose)
      cerr << "dsp::FoldManager::combine other is not FoldManager" << endl;
    return;
  }

  if (manager->fold.size() != fold.size())
    throw Error (InvalidState, "dsp::FoldManager::combine",
                 "other nfold=%u != this fold=%u", 
                 manager->fold.size(), fold.size());

  for (unsigned i=0; i<fold.size(); i++)
  {
    if (verbose)
      cerr << "dsp::FoldManager::combine fold[" << i << "]=" << (void*) fold[i].get()
           << " other[" << i << "]=" << (void*) manager->fold[i].get() << endl;

    fold[i]->combine (manager->fold[i]);
  }
}

void dsp::FoldManager::reset ()
{
  for (unsigned i=0; i<fold.size(); i++)
    fold[i]->reset ();
}

void dsp::FoldManager::finish ()
{
  for (unsigned i=0; i<fold.size(); i++)
    fold[i]->finish();
}

void dsp::FoldManager::set_cerr (std::ostream& os) const
{
  Operation::set_cerr (os);
  for (unsigned i=0; i<fold.size(); i++)
    fold[i]->set_cerr (os);
}

