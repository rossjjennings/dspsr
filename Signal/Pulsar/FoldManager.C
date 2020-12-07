/***************************************************************************
 *
 *   Copyright (C) 2020 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/FoldManager.h"
#include "dsp/Fold.h"

dsp::FoldManager::FoldManager () : Operation("FoldManager")
{
}

void dsp::FoldManager::manage (Fold* a_fold)
{
  fold.push_back (a_fold);
}

void dsp::FoldManager::operation()
{
  for (unsigned i=0; i<fold.size(); i++)
    fold[i]->operate();
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
  const FoldManager* manager = dynamic_cast<const FoldManager*> (other);
  if (!manager)
    return;

  if (manager->fold.size() != fold.size())
    throw Error (InvalidState, "dsp::FoldManager::combine",
                 "other nfold=%u != this fold=%u", 
                 manager->fold.size(), fold.size());

  for (unsigned i=0; i<fold.size(); i++)
    fold[i]->combine (manager->fold[i]);
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


