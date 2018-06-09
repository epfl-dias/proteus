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

#include "common/common.hpp"

double
diff(struct timespec st, struct timespec end)
{
	struct timespec tmp;

	if ((end.tv_nsec-st.tv_nsec)<0) {
		tmp.tv_sec = end.tv_sec - st.tv_sec - 1;
		tmp.tv_nsec = 1e9 + end.tv_nsec - st.tv_nsec;
	} else {
		tmp.tv_sec = end.tv_sec - st.tv_sec;
		tmp.tv_nsec = end.tv_nsec - st.tv_nsec;
	}

	return tmp.tv_sec + tmp.tv_nsec * 1e-9;
}

void
fatal(const char *err)
{
    perror(err);
    exit(1);
}

void
exception(const char *err)
{
    printf("Exception: %s\n", err);
    exit(1);
}

size_t getFileSize(const char* filename) {
    struct stat st;
    stat(filename, &st);
    return st.st_size;
}


std::ostream& operator<<(std::ostream& out, const bytes& b){
	const char *units[]{"B", "KB", "MB", "GB", "TB", "ZB"};
	constexpr size_t max_i = sizeof(units) / sizeof(units[0]);
	size_t bs = b.b * 10;

	size_t i = 0;
	while (bs >= 10240 && i < max_i) {
		bs /= 1024;
		++i;
	}

	out << (bs/10.0) << units[i];
	return out;
}

int get_device(const void *p){
#ifndef NCUDA
    cudaPointerAttributes attrs;
    cudaError_t error = cudaPointerGetAttributes(&attrs, p);
    if (error == cudaErrorInvalidValue) return -1;
    gpu(error);
    return (attrs.memoryType == cudaMemoryTypeHost) ? -1 : attrs.device;
#else
    return -1;
#endif
}
