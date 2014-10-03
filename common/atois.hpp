/*
	RAW -- High-performance querying over raw, never-seen-before data.

							Copyright (c) 2014
		Data Intensive Applications and Systems Labaratory (DIAS)
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

#ifndef ATOIS_HPP_
#define ATOIS_HPP_

#include "common/common.hpp"

inline int atoi1(const char *buf) {
	return  (buf[0] - '0');
}

inline int atoi2(const char *buf) {
	return  ((buf[0] - '0') * 10) + \
			(buf[1] - '0');
}

inline int atoi3(const char *buf) {
	return  ((buf[0] - '0') * 100) + \
			((buf[1] - '0') * 10) + \
			(buf[2] - '0');
}

inline int atoi4(const char *buf) {
	return  ((buf[0] - '0') * 1000) + \
			((buf[1] - '0') * 100) + \
			((buf[2] - '0') * 10) + \
			(buf[3] - '0');
}

inline int atoi5(const char *buf) {
	return  ((buf[0] - '0') * 10000) + \
			((buf[1] - '0') * 1000) + \
			((buf[2] - '0') * 100) + \
			((buf[3] - '0') * 10) + \
			(buf[4] - '0');
}

inline int atoi6(const char *buf) {
	return  ((buf[0] - '0') * 100000) + \
			((buf[1] - '0') * 10000) + \
			((buf[2] - '0') * 1000) + \
			((buf[3] - '0') * 100) + \
			((buf[4] - '0') * 10) + \
			(buf[5] - '0');
}

inline int atoi7(const char *buf) {
	return  ((buf[0] - '0') * 1000000) + \
			((buf[1] - '0') * 100000) + \
			((buf[2] - '0') * 10000) + \
			((buf[3] - '0') * 1000) + \
			((buf[4] - '0') * 100) + \
			((buf[5] - '0') * 10) + \
			(buf[6] - '0');
}

inline int atoi8(const char *buf) {
	return  ((buf[0] - '0') * 10000000) + \
			((buf[1] - '0') * 1000000) + \
			((buf[2] - '0') * 100000) + \
			((buf[3] - '0') * 10000) + \
			((buf[4] - '0') * 1000) + \
			((buf[5] - '0') * 100) + \
			((buf[6] - '0') * 10) + \
			(buf[7] - '0');
}

inline int atoi9(const char *buf) {
	return  ((buf[0] - '0') * 100000000) + \
			((buf[1] - '0') * 10000000) + \
			((buf[2] - '0') * 1000000) + \
			((buf[3] - '0') * 100000) + \
			((buf[4] - '0') * 10000) + \
			((buf[5] - '0') * 1000) + \
			((buf[6] - '0') * 100) + \
			((buf[7] - '0') * 10) + \
			(buf[8] - '0');
}

inline int atoi10(const char *buf) {
	return  ((buf[0] - '0') * 1000000000) + \
			((buf[1] - '0') * 100000000) + \
			((buf[2] - '0') * 10000000) + \
			((buf[3] - '0') * 1000000) + \
			((buf[4] - '0') * 100000) + \
			((buf[5] - '0') * 10000) + \
			((buf[6] - '0') * 1000) + \
			((buf[7] - '0') * 100) + \
			((buf[8] - '0') * 10) + \
			(buf[9] - '0');
}

inline int atois(const char *buf, int len) {
	switch (len) {
	case 1:
		return atoi1(buf);
	case 2:
		return atoi2(buf);
	case 3:
		return atoi3(buf);
	case 4:
		return atoi4(buf);
	case 5:
		return atoi5(buf);
	case 6:
		return atoi6(buf);
	case 7:
		return atoi7(buf);
	case 8:
		return atoi8(buf);
	case 9:
		return atoi9(buf);
	case 10:
		return atoi10(buf);
	default:
		LOG(ERROR) << "[ATOIS: ] Invalid Size " << len;
		throw runtime_error(string("[ATOIS: ] Invalid Size "));
	}
}

#endif /* ATOIS_HPP_ */
