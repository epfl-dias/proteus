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

bool verifyTestResult(const char *testsPath, const char *testLabel)	{
	/* Compare with template answer */
	/* correct */
	struct stat statbuf;
	string correctResult = string(testsPath) + testLabel;
	stat(correctResult.c_str(), &statbuf);
	size_t fsize1 = statbuf.st_size;
	int fd1 = open(correctResult.c_str(), O_RDONLY);
	if (fd1 == -1) {
		throw runtime_error(string(__func__) + string(".open (verification): ")+correctResult);
	}
	char *correctBuf = (char*) mmap(NULL, fsize1, PROT_READ | PROT_WRITE,
			MAP_PRIVATE, fd1, 0);

	/* current */
	stat(testLabel, &statbuf);
	size_t fsize2 = statbuf.st_size;
	int fd2 = shm_open(testLabel, O_RDONLY, S_IRWXU);
	if (fd2 == -1) {
		throw runtime_error(string(__func__) + string(".open (output): ")+testLabel);
	}
	char *currResultBuf = (char*) mmap(NULL, fsize2, PROT_READ | PROT_WRITE,
			MAP_PRIVATE, fd2, 0);
	bool areEqual = ((fsize1 == fsize2) && ((fsize1 == 0) || (memcmp(correctBuf, currResultBuf, fsize1) == 0))) ? true : false;
	if (!areEqual) {
		fprintf(stderr, "######################################################################\n");
		fprintf(stderr, "FAILURE:\n");
		if (fsize1 > 0) fprintf(stderr, "* Expected (size: %zu):\n%s\n", fsize1, correctBuf);
		else 			fprintf(stderr, "* Expected empty file\n");
		if (fsize2 > 0) fprintf(stderr, "* Obtained (size: %zu):\n%s\n", fsize2, currResultBuf);
		else 			fprintf(stderr, "* Obtained empty file\n");
		fprintf(stderr, "######################################################################\n");
	}

	close(fd1);
	munmap(correctBuf, fsize1);
	// close(fd2);
	shm_unlink(testLabel);
	munmap(currResultBuf, fsize2);
	// if (remove(testLabel) != 0) {
	// 	throw runtime_error(string("Error deleting file"));
	// }

	return areEqual;
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
