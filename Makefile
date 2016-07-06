# Makefile for pipeline_engine
#
# The following environment variables should be defined
#
# CILKPLUS_PREFIX - Location of the CILK install
#
# OPENCV_PREFIX - Location of the install of OpenCV 2.4
#
# VIGRA_PREFIX - Location of the install of Vigra
#

# from http://stackoverflow.com/questions/10858261/abort-makefile-if-variable-not-set
check_defined = \
	$(foreach 1,$1,$(__check_defined))
__check_defined = \
	$(if $(value $1),,$(error Undefined $1$(if $(value 2), ($(strip $2)))))

$(call check_defined, CILKPLUS_PREFIX, CILK parallel compiler root directory)
$(call check_defined, VIGRA_PREFIX, Vigra root directory)
$(call check_defined, OPENCV_PREFIX, OpenCV)

export OPENCV_INCLUDE=$(OPENCV_PREFIX)/include
export OPENCV_LIB=$(OPENCV_PREFIX)/lib

.PHONY: all watershed np-merge

all: watershed np-merge

watershed:
	cd watershed &&\
	make

np-merge:
	cd merge/np-merge/cpp &&\
	make