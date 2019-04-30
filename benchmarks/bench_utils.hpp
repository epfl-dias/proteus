/*
                  AEOLUS - In-Memory HTAP-Ready OLTP Engine

                              Copyright (c) 2019-2019
           Data Intensive Applications and Systems Laboratory (DIAS)
                   Ecole Polytechnique Federale de Lausanne

                              All Rights Reserved.

      Permission to use, copy, modify and distribute this software and its
    documentation is hereby granted, provided that both the copyright notice
  and this permission notice appear in all copies of the software, derivative
  works or modified versions, and any portions thereof, and that both notices
                      appear in supporting documentation.

  This code is distributed in the hope that it will be useful, but WITHOUT ANY
 WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 A PARTICULAR PURPOSE. THE AUTHORS AND ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE
DISCLAIM ANY LIABILITY OF ANY KIND FOR ANY DAMAGES WHATSOEVER RESULTING FROM THE
                             USE OF THIS SOFTWARE.
*/

#ifndef BENCH_UTILS_HPP_
#define BENCH_UTILS_HPP_

#include <cassert>
#include <cmath>
#include <cstdlib>

namespace bench {

#define FALSE 0
#define TRUE 1

static int RAND(unsigned int *seed, int max) { return rand_r(seed) % max; }

static int URand(unsigned int *seed, int x, int y) {
  return x + RAND(seed, y - x + 1);
}

static int NURand(unsigned int *seed, int A, int x, int y) {
  static char C_255_init = FALSE;
  static char C_1023_init = FALSE;
  static char C_8191_init = FALSE;
  static int C_255, C_1023, C_8191;
  int C = 0;
  switch (A) {
    case 255:
      if (!C_255_init) {
        C_255 = URand(seed, 0, 255);
        C_255_init = TRUE;
      }
      C = C_255;
      break;
    case 1023:
      if (!C_1023_init) {
        C_1023 = URand(seed, 0, 1023);
        C_1023_init = TRUE;
      }
      C = C_1023;
      break;
    case 8191:
      if (!C_8191_init) {
        C_8191 = URand(seed, 0, 8191);
        C_8191_init = TRUE;
      }
      C = C_8191;
      break;
    default:
      assert(0);
      exit(-1);
  }
  return (((URand(seed, 0, A) | URand(seed, x, y)) + C) % (y - x + 1)) + x;
}

static int make_alpha_string(unsigned int *seed, int min, int max, char *str) {
  char char_list[] = {'1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b',
                      'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
                      'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x',
                      'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
                      'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                      'U', 'V', 'W', 'X', 'Y', 'Z'};
  int cnt = URand(seed, min, max);
  for (uint32_t i = 0; i < cnt; i++) str[i] = char_list[URand(seed, 0L, 60L)];

  for (int i = cnt; i < max; i++) str[i] = '\0';

  return cnt;
}

static int make_numeric_string(unsigned int *seed, int min, int max,
                               char *str) {
  int cnt = URand(seed, min, max);

  for (int i = 0; i < cnt; i++) {
    int r = URand(seed, 0L, 9L);
    str[i] = '0' + r;
  }
  return cnt;
}

}  // namespace bench

#endif /* BENCH_UTILS_HPP_ */
