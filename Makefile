#LIBJIT_PATH = /usr/local/lib/
#LIBJIT_INCLUDE_PATH = $(LIBJIT_PATH)/include
#LIBJIT_LIB_PATH = $(LIBJIT_PATH)/jit/.libs
#LIBJIT_AR = $(LIBJIT_LIB_PATH)/libjit.a

CC = clang #gcc
CPP = clang++ #g++ 

CCOPT = -ggdb -O0 -fkeep-inline-functions
#-O3
CCFLAGS = -c $(CCOPT) 

#Assumes gtest has been built in the home directory of the user
USER_DIR = $(HOME)
GTEST_DIR = $(USER_DIR)/gtest-1.7.0

LLVMOPT =`llvm-config --cppflags`
LLVMJIT =`llvm-config --cppflags --ldflags --libs core jit native` -rdynamic
LDFLAGS = -L$(HOME)/lib -L. -lglog -ljsmn \
		-lboost_system \
		#-lboost_iostreams -lboost_thread -lboost_filesystem \
		

#LDFLAGS = -lpthread -lm -ldl -ljit -lrt

#REMINDER:
#$< : first dependecy
#$^ : all dependencies


# Flags passed to the preprocessor.
# Set Google Test's header directory as a system directory, such that
# the compiler doesn't generate warnings in Google Test headers.
# -isystem $(GTEST_DIR)/include are relevant for the google testing framework
CPPFLAGS = -I$(HOME)/include -I. -I./util/ -I./plugins/ -I./jsmn/ \
			-isystem $(GTEST_DIR)/include -I$(GTEST_DIR) \
			#-std=c++11 \

# Flags passed to the C++ compiler.
CXXFLAGS += -g -pthread

# All Google Test headers.  Usually you shouldn't change this
# definition.
GTEST_HEADERS = $(GTEST_DIR)/include/gtest/*.h \
                $(GTEST_DIR)/include/gtest/internal/*.h

RMOBJS = main util/raw-catalog.o util/raw-context.o plugins/object/plugins-llvm.o expressions/binary-operators.o \
		 expressions/expressions.o  expressions/expressions-generator.o values/expressionTypes.o  \
		 expressions/path.o \
		 operators/scan.o operators/select.o operators/join.o operators/print.o operators/root.o \
		 operators/unnest.o operators/outer-unnest.o operators/reduce.o \
		 plugins/csv-plugin.o plugins/binary-row-plugin.o \
		 plugins/json-jsmn-plugin.o \
		 plugins/output/plugins-output.o plugins/helpers.o common/common.o tests/test1.o tests-relational.o gtest_main.o gtest-all.o \
		 tests-operators.o tests-sailors.o benchmark tests-sailors tests-operators gtest_main.a \
		 jsmn/jsmn.o libjmn.a 

all: main

#header file code
common/common.o: common/common.cpp
	${CPP} ${CCFLAGS} ${CPPFLAGS} $(CXXFLAGS) ${LLVMOPT} $^ -o $@

values/expressionTypes.o: values/expressionTypes.cpp
	${CPP} ${CCFLAGS} ${CPPFLAGS} $(CXXFLAGS) ${LLVMOPT} $^ -o $@

plugins/csv-plugin.o: plugins/csv-plugin.cpp
	${CPP} ${CCFLAGS} ${CPPFLAGS} $(CXXFLAGS) ${LLVMOPT} $^ -o $@

plugins/binary-row-plugin.o: plugins/binary-row-plugin.cpp
	${CPP} ${CCFLAGS} ${CPPFLAGS} $(CXXFLAGS) ${LLVMOPT} $^ -o $@

plugins/json-jsmn-plugin.o: plugins/json-jsmn-plugin.cpp
	${CPP} ${CCFLAGS} ${CPPFLAGS} $(CXXFLAGS) ${LLVMOPT} $^ -o $@

plugins/output/plugins-output.o: plugins/output/plugins-output.cpp
	${CPP} ${CCFLAGS} ${CPPFLAGS} $(CXXFLAGS) ${LLVMOPT} $^ -o $@

util/raw-context.o: util/raw-context.cpp
	${CPP} ${CCFLAGS} ${CPPFLAGS} $(CXXFLAGS) ${LLVMOPT} $^ -o $@

util/raw-catalog.o: util/raw-catalog.cpp
	${CPP} ${CCFLAGS} ${CPPFLAGS} $(CXXFLAGS) ${LLVMOPT} $^ -o $@

expressions/binary-operators.o: expressions/binary-operators.cpp
	${CPP} ${CCFLAGS} ${CPPFLAGS} $(CXXFLAGS) ${LLVMOPT} $^ -o $@

expressions/expressions.o: expressions/expressions.cpp
	${CPP} ${CCFLAGS} ${CPPFLAGS} $(CXXFLAGS) ${LLVMOPT} $^ -o $@

expressions/expressions-generator.o: expressions/expressions-generator.cpp
	${CPP} ${CCFLAGS} ${CPPFLAGS} $(CXXFLAGS) ${LLVMOPT} $^ -o $@

expressions/path.o: expressions/path.cpp
	${CPP} ${CCFLAGS} ${CPPFLAGS} $(CXXFLAGS) ${LLVMOPT} $^ -o $@

operators/scan.o: operators/scan.cpp
	${CPP} ${CCFLAGS} ${CPPFLAGS} $(CXXFLAGS) ${LLVMOPT} $^ -o $@

operators/select.o: operators/select.cpp
	${CPP} ${CCFLAGS} ${CPPFLAGS} $(CXXFLAGS) ${LLVMOPT} $^ -o $@

operators/join.o: operators/join.cpp
	${CPP} ${CCFLAGS} ${CPPFLAGS} $(CXXFLAGS) ${LLVMOPT} $^ -o $@

operators/unnest.o: operators/unnest.cpp
	${CPP} ${CCFLAGS} ${CPPFLAGS} $(CXXFLAGS) ${LLVMOPT} $^ -o $@

operators/outer-unnest.o: operators/outer-unnest.cpp
	${CPP} ${CCFLAGS} ${CPPFLAGS} $(CXXFLAGS) ${LLVMOPT} $^ -o $@

operators/reduce.o: operators/reduce.cpp
	${CPP} ${CCFLAGS} ${CPPFLAGS} $(CXXFLAGS) ${LLVMOPT} $^ -o $@

operators/print.o: operators/print.cpp
	${CPP} ${CCFLAGS} ${CPPFLAGS} $(CXXFLAGS) ${LLVMOPT} $^ -o $@

operators/root.o: operators/root.cpp
	${CPP} ${CCFLAGS} ${CPPFLAGS} $(CXXFLAGS) ${LLVMOPT} $^ -o $@

libjsmn.a: jsmn/jsmn.o
	$(AR) rc $@ $^

jsmn/jsmn.o: jsmn/jsmn.c jsmn/jsmn.h
	$(CC) -c $(CFLAGS) $< -o $@

main: main.cpp libjsmn.a common/common.o values/expressionTypes.o \
	plugins/csv-plugin.o plugins/binary-row-plugin.o \
	plugins/json-jsmn-plugin.o plugins/output/plugins-output.o \
	operators/scan.o operators/select.o operators/join.o operators/print.o operators/root.o operators/unnest.o \
	operators/reduce.o operators/outer-unnest.o \
	util/raw-catalog.o util/raw-context.o \
	expressions/binary-operators.o expressions/expressions.o expressions/expressions-generator.o \
	expressions/path.o
	${CPP} ${CCOPT} ${CPPFLAGS} $(CXXFLAGS) $^ ${LLVMJIT} -o $@ ${LDFLAGS}

clean:
	rm -rf ${RMOBJS}


# Builds gtest.a and gtest_main.a.

# Usually you shouldn't tweak such internal variables, indicated by a
# trailing _.
GTEST_SRCS_ = $(GTEST_DIR)/src/*.cc $(GTEST_DIR)/src/*.h $(GTEST_HEADERS)

# For simplicity and to avoid depending on Google Test's
# implementation details, the dependencies specified below are
# conservative and not optimized.  This is fine as Google Test
# compiles fast and for ordinary users its source rarely changes.
gtest-all.o : $(GTEST_SRCS_)
	$(CPP) $(CPPFLAGS) -I$(GTEST_DIR) $(CXXFLAGS) -c \
            $(GTEST_DIR)/src/gtest-all.cc

gtest_main.o : gtest_main.cpp 
	$(CPP) $(CPPFLAGS) -I$(GTEST_DIR) $(CXXFLAGS) -c $<



gtest.a : gtest-all.o
	$(AR) $(ARFLAGS) $@ $^

gtest_main.a : gtest-all.o gtest_main.o
	$(AR) $(ARFLAGS) $@ $^

# Builds a sample test.  A test should link with either gtest.a or
# gtest_main.a, depending on whether it defines its own main()
# function.

#sample1.o : $(USER_DIR)/sample1.cc $(USER_DIR)/sample1.h $(GTEST_HEADERS)
#	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $(USER_DIR)/sample1.cc

tests-operators.o : tests/tests-operators.cpp $(GTEST_HEADERS) 
	$(CPP) $(CPPFLAGS) $(CXXFLAGS) ${LLVMOPT} -c tests/tests-operators.cpp 

tests-operators : tests-operators.o gtest_main.a libjsmn.a common/common.o values/expressionTypes.o \
					plugins/csv-plugin.o plugins/binary-row-plugin.o \
					plugins/json-jsmn-plugin.o \
					plugins/output/plugins-output.o operators/scan.o operators/select.o operators/join.o operators/print.o \
					operators/reduce.o operators/outer-unnest.o \
					operators/root.o operators/unnest.o util/raw-catalog.o util/raw-context.o expressions/binary-operators.o \
					expressions/expressions.o  expressions/expressions-generator.o expressions/path.o
	$(CPP) $(CPPFLAGS) $(CXXFLAGS) -lpthread $^ ${LLVMJIT} -o $@ ${LDFLAGS}

tests-sailors.o : tests/tests-sailors.cpp $(GTEST_HEADERS)
	$(CPP) $(CPPFLAGS) $(CXXFLAGS) ${LLVMOPT} -c tests/tests-sailors.cpp 

tests-sailors : tests-sailors.o gtest_main.a libjsmn.a common/common.o values/expressionTypes.o \
				plugins/csv-plugin.o plugins/binary-row-plugin.o \
				plugins/json-jsmn-plugin.o \
				plugins/output/plugins-output.o operators/scan.o operators/select.o operators/join.o \
				operators/reduce.o operators/outer-unnest.o \
				operators/print.o operators/root.o operators/unnest.o util/raw-catalog.o util/raw-context.o \
				expressions/binary-operators.o expressions/expressions.o  expressions/expressions-generator.o \
				expressions/path.o
	$(CPP) $(CPPFLAGS) $(CXXFLAGS) -lpthread $^ ${LLVMJIT} -o $@ ${LDFLAGS}