// don't forget to use appropriate sourceCpp("path/to/file.cpp") from R to add a function to working environment
#include <Rcpp.h>

using namespace Rcpp;

// [[Rcpp::export]]
XPtr<std::vector<int> > initVector(int size) {
  std::vector<int>* v = new std::vector<int>;

  for(int i=0; i<size; i++) {
    v->push_back(i);
  }

  XPtr<std::vector<int> > p(v, true);

  return(p);
}

int* initArr(int size) {
  int * arr = new int[size];

  for(int i=0; i<size; i++) {
    arr[i] = i;
  }

  return(arr);
}

// [[Rcpp::export]]
DataFrame readBlocks() {
  // access the columns
  // hopefully some metadata to know what columns to create
  IntegerVector a = IntegerVector::create();
  CharacterVector b = CharacterVector::create();
  RawVector c = RawVector::create();

  // return a new data frame
  return DataFrame::create(_["a"]= a, _["b"]= b, _["c"]=c);
}

// [[Rcpp::export]]
DataFrame readFromXptr(XPtr<std::vector<int> > vec_ptr) {
  Rcpp::Rcout << vec_ptr->at(50);

  // TODO: need to figure out how the Rcpp _Vectors can use pointers instead of copies
  // it is easy to read directly, BUT with the cost of allocating new Vector for R
  // also, the object has to be protected from R GC
  // one "solution" is to have an array allocating some memory here, reading from shm and then dealocating
  // we again read and write, and consume 2x memory at worst point

  // Still researching the option to directly point to values in memory
  // it seems that this is a fundamental limitation


  IntegerVector iv(vec_ptr->begin(), vec_ptr->end());

  return DataFrame::create(_["a"]= iv);
  //return 1;
}

// [[Rcpp::export]]
DataFrame readFromArr(int size) {
  int* arr = initArr(size);

  Rcpp:Rcout << arr[20];
  SEXP iv = coerceVector(arr, INTSXP);

  //Rcpp:Rcout << typeid(iv).name();

  //return DataFrame::create(_["a"]= iv);
  return 1;
}

// [[Rcpp::export]]
bool freeMemory() {
  return true;
 }
