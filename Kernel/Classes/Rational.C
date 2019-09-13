/***************************************************************************
 *
 *   Copyright (C) 2018 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "Rational.h"
#include <stdlib.h>
#include <stdexcept>

std::istream& operator >> (std::istream& in, Rational& r)
{
  char divide = 0;
  in >> r.numerator >> divide >> r.denominator;
  if (divide != '/')
    in.setstate(std::istream::failbit);

  return in;
}

std::ostream& operator << (std::ostream& out, const Rational& r)
{
  out << r.numerator << "/" << r.denominator;
  return out;
}


Rational::Rational (int num, int den)
{
  numerator = num;
  denominator = den;
}

const Rational& Rational::operator = (const Rational& r)
{
  numerator = r.numerator;
  denominator = r.denominator;
  return *this;
}

bool Rational::operator == (const Rational& r) const
{
  return numerator == r.numerator && denominator == r.denominator;
}

bool Rational::operator != (const Rational& r) const
{
  return numerator != r.numerator || denominator != r.denominator;
}

const Rational& Rational::operator = (int num)
{
  numerator = num;
  denominator = 1;
  return *this;
}

bool Rational::operator == (int num) const
{
  return denominator == 1 && numerator == num;
}

bool Rational::operator != (int num) const
{
  return denominator != 1 || numerator != num;
}

int Rational::operator * (int num) const
{
  div_t result = div(num * numerator, denominator);
  if (result.rem != 0) {
    throw std::domain_error("Rational operator / result is not an integer");
  }
  return result.quot;
}

double Rational::doubleValue( ) const
{
  return double(numerator) / double(denominator);
}

int Rational::normalize (int i) const
{
  div_t result = div (i * denominator, numerator);
  if (result.rem != 0)
    throw std::domain_error ("Rational operator / result is not an integer");

  return result.quot;
}

// double Rational::normalize (double d) const
// {
//   return d * denominator / numerator;
// }
