CC=g++
CXXFLAGS=-c -fPIC -Wall -Werror -Wextra -Wpedantic -std=c++17 `pkg-config --cflags Qt5Gui`
LDFLAGS=`pkg-config --libs Qt5Gui` -lgtest -lm -pthread
TESTSRC=./tests/tests.cc
SRCFILES=model/*.cc model/graphnetwork/*.cc model/matrixnetwork/*.cc model/interfaces/*.h model/storage/*.cc model/utils/*.cc view/*.cc view/canvas/canvas.cc controller/*.cc
HDRFILES=$(SRCFILES:.cc=.h)
TESTOBJ=$(TESTSRC:.cc=.o)
LIBOBJ=$(LIBSRC:.cc=.o)
PROJECTNAME=CPP7_MLP-0
EXECUTABLE=./tests/$(PROJECTNAME)_tests
INSTALLDIR=build
LCOVEXEC=$(EXECUTABLE).info
REPORTDIR=report
LEAKS_REPORT_FILE=leaks_report.txt
UNAME_S=$(shell uname -s)
ifeq ($(UNAME_S), Linux)
LEAK_TEST=valgrind --leak-check=full --verbose --log-file=$(REPORTDIR)/$(LEAKS_REPORT_FILE)
else
LEAK_TEST=CK_FORK=no leaks -atExit -- 
endif

.PHONY: all install uninstall clean dvi dist tests build_tests style cppcheck gcov_report

all: install

%.o: %.cc
	$(CC) $(CXXFLAGS) $^ -o $@

install: uninstall
	@cmake -S .. -B $(INSTALLDIR) && cmake --build $(INSTALLDIR) 

uninstall:
	@rm -rf $(INSTALLDIR)

clean:
	@rm -rf ./model/*.o ./model/*.gcno ./model/*.gcda ./model/*.gcov \
	./tests/*.o ./tests/*.gcno ./tests/*.gcda ./tests/*.gcov \
	$(EXECUTABLE) $(LCOVEXEC) $(REPORTDIR) $(INSTALLDIR) $(DVI_FILE).pdf

dvi:
ifeq ($(UNAME_S), Linux)
	xdg-open Description.md
endif
ifeq ($(UNAME_S), Darwin)
	open Description.md
endif

dist: install
	@cd $(INSTALLDIR) && tar -czf $(PROJECTNAME).tgz $(PROJECTNAME) && rm -f $(PROJECTNAME)

build_tests: clean $(TESTOBJ) $(LIBOBJ)
	$(CC) $(TESTOBJ) $(LIBOBJ) -o $(EXECUTABLE) $(LDFLAGS)

tests: build_tests
	./$(EXECUTABLE)

leaks: build_tests
	$(LEAK_TEST) ./$(EXECUTABLE)

style:
	clang-format -n -style=google $(HDRFILES) $(SRCFILES) main.cc

cppcheck:
	cppcheck --language=c++ --std=c++17 --enable=all --suppress=*:view/canvas/qcustomplot.* --suppress=unusedFunction \
	--suppress=missingInclude --suppress=unusedStructMember $(SRCFILES) $(HDRFILES) main.cc

gcov_report: CXXFLAGS+=--coverage
gcov_report: LDFLAGS+=--coverage
gcov_report: tests
	lcov -t $(EXECUTABLE) -o $(LCOVEXEC) --include */src/model/*.cc -c -d ./tests 
	genhtml -o $(REPORTDIR) $(LCOVEXEC)
