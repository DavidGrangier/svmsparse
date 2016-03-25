# SVM with stochastic gradient

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111, USA


L=sgd/lib
P=sgd/svm
CXX=g++
OPT=
OPTS=-g -O2
CXXFLAGS= ${OPTS} ${OPT} -Wall -I$L -I$P
LIBS = -lz -lm

PROGRAMS = libsvmsparse.so
OBJS = vectors.o
INCS = $L/vectors.h $L/wrapper.h $L/assert.h

all: ${PROGRAMS}

clean:
	-rm ${PROGRAMS} 2>/dev/null
	-rm *.o 2>/dev/null

vectors.o: $L/vectors.cpp ${INCS}
	${CXX} ${CXXFLAGS} -fPIC -c -o $@ $L/vectors.cpp

svmsgd.o: svmsgd.cpp svmsgd.h $P/loss.h ${INCS}
	${CXX} ${CXXFLAGS} -fPIC -c -o $@ svmsgd.cpp

svmsparse.o: svmsparse.cpp $P/loss.h ${INCS}
	${CXX} $(CXXFLAGS) -fPIC -c -o $@ svmsparse.cpp
	
libsvmsparse.so: svmsparse.o svmsgd.o ${OBJS}	
	${CXX} -shared -o $@ svmsparse.o svmsgd.o ${OBJS} ${LIBS}

install: libsvmsparse.so svmsparse.lua 
	cp libsvmsparse.so $(INST_LIBDIR)
	mkdir -p $(INST_LUADIR)/svmsparse/
	cp svmsparse.lua $(INST_LUADIR)/svmsparse/init.lua
