# FlatFieldApp Makefile

CC=gcc

CFLAGS= -O2 -pipe -fPIC -W -Wextra -ggdb
CFLAGS+= -I../include

#LDFLAGS= -Wl,-rpath,$(HOME)/meteor/lib
LDFLAGS= -lm

LIBTRAJECTORYSOLUTION_OBJS=TrajectorySolution.o
OBJS=$(LIBTRAJECTORYSOLUTION_OBJS)

INCL=TrajectorySolution.h
LIBS=libtrajectorysolution.so libtrajectorysolution.so.0
BIN=

default:	shared

shared:		$(LIBTRAJECTORYSOLUTION_OBJS)
		$(CC) $(LDFLAGS) -shared -Wl,-soname,libtrajectorysolution.so -o libtrajectorysolution.so.0 $(LIBTRAJECTORYSOLUTION_OBJS)
		ln -sf libtrajectorysolution.so.0 libtrajectorysolution.so

%.o:		%.c %.h
		$(CC) $(CFLAGS) -o $@ -c $<

clean:
		rm -f $(OBJS) $(LIBS) $(BIN)

install:
		install -m 0755 -d $(HOME)/meteor/lib
		install -m 0755 $(LIBS) $(HOME)/meteor/lib
		install -m 0755 -d $(HOME)/meteor/include/trajectorysolution
		install -m 0755 $(INCL) $(HOME)/meteor/include/trajectorysolution
