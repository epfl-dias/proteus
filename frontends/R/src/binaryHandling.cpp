/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2018
        Data Intensive Applications and Systems Laboratory (DIAS)
                École Polytechnique Fédérale de Lausanne

                            All Rights Reserved.

    Permission to use, copy, modify and distribute this software and
    its documentation is hereby granted, provided that both the
    copyright notice and this permission notice appear in all copies of
    the software, derivative works or modified versions, and any
    portions thereof, and that both notices appear in supporting
    documentation.

    This code is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. THE AUTHORS
    DISCLAIM ANY LIABILITY OF ANY KIND FOR ANY DAMAGES WHATSOEVER
    RESULTING FROM THE USE OF THIS SOFTWARE.
*/
// don't forget to use appropriate sourceCpp("path/to/file.cpp") from R to add a
// function to working environment
#include <R.h>
#include <Rcpp.h>
#include <Rinternals.h>

using namespace Rcpp;

// [[Rcpp::export]]
XPtr<std::vector<int> > initVector(int size) {
  std::vector<int>* v = new std::vector<int>;

  for (int i = 0; i < size; i++) {
    v->push_back(i);
  }

  XPtr<std::vector<int> > p(v, true);

  return (p);
}

int* initArr(int size) {
  int* arr = new int[size];

  for (int i = 0; i < size; i++) {
    arr[i] = i;
  }

  return (arr);
}

// [[Rcpp::export]]
DataFrame readBlocks() {
  // access the columns
  // hopefully some metadata to know what columns to create
  IntegerVector a = IntegerVector::create();
  CharacterVector b = CharacterVector::create();
  RawVector c = RawVector::create();

  // return a new data frame
  return DataFrame::create(_["a"] = a, _["b"] = b, _["c"] = c);
}

// [[Rcpp::export]]
DataFrame readFromXptr(XPtr<std::vector<int> > vec_ptr) {
  Rcpp::Rcout << vec_ptr->at(50);

  // TODO: need to figure out how the Rcpp _Vectors can use pointers instead of
  // copies it is easy to read directly, BUT with the cost of allocating new
  // Vector for R also, the object has to be protected from R GC one "solution"
  // is to have an array allocating some memory here, reading from shm and then
  // dealocating we again read and write, and consume 2x memory at worst point

  // Still researching the option to directly point to values in memory
  // it seems that this is a fundamental limitation

  // std::vector<int> iv =

  IntegerVector iv(vec_ptr->begin(), vec_ptr->end());

  return DataFrame::create(_["a"] = iv);
  // return 1;
}

// [[Rcpp::export]]
DataFrame readTest(int size) {
  XPtr<std::vector<int> > vec_ptr = initVector(size);
  Rcpp::Rcout << vec_ptr->at(50);

  IntegerVector iv(vec_ptr->begin(), vec_ptr->end());

  return DataFrame::create(_["a"] = iv);
  // return 1;
}

// [[Rcpp::export]]
DataFrame readFromArr(int size) {
  int* arr = initArr(size);

Rcpp:
  Rcout << arr[20];

  // Rcpp:Rcout << sizeof(INTEGER);

  // return DataFrame::create(_["a"]= iv);
  return 1;
}

// [[Rcpp::export]]
DataFrame test(int size) {
  // This is in theory what should be working
  // We create a c++ vector from the shm
  // R will create a copy and we then deallocate
  std::vector<int>* v = new std::vector<int>();

  for (int i = 0; i < size; i++) {
    v->push_back(1234);
  }

  // NumericVector xx(y.begin(), y.end());

  // Rcpp:Rcout << sizeof(xx) << ", " << sizeof(y);

  DataFrame df = DataFrame::create(_["a"] = *v);

  v->clear();
  v = new std::vector<int>();

  return df;
}

// [[Rcpp::export]]
bool freeMemory() { return true; }
