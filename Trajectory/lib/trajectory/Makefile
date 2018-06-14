# libtrajectorysolution Makefile
have_conf	:= $(wildcard $(HOME)/meteor/etc/trajectorysolution.conf)

CC=gcc

CFLAGS= -O2 -pipe -fPIC -W -Wextra -ggdb
CFLAGS+= -I../include

#LDFLAGS= -Wl,-rpath,$(HOME)/meteor/lib
LDFLAGS= -lm

LIBTS_BUILD=TrajectorySolution.o
LIBTS_SO=libtrajectorysolution.so.0
LIBTS_SONAME=libtrajectorysolution.so
LIBTS_OBJS=$(LIBTS_BUILD)
LIBTS_OBJS+=../common/ParticleSwarmFunctions.o
LIBTS_INCL=TrajectorySolution.h

ALL_OBJS=$(LIBTS_BUILD)
ALL_INCL=$(LIBTS_INCL)
ALL_LIBS=$(LIBTS_SO)
ALL_BIN=

INCL=TrajectorySolution.h
LIBS=libtrajectorysolution.so libtrajectorysolution.so.0
BIN=

default:	shared

shared:		$(LIBTS_BUILD)
		$(CC) $(LDFLAGS) -shared -Wl,-soname,$(LIBTS_SONAME) -o $(LIBTS_SO) \
			$(LIBTS_OBJS)
		ln -sf libtrajectorysolution.so.0 libtrajectorysolution.so

%.o:		%.c %.h
		$(CC) $(CFLAGS) -o $@ -c $<

clean:
		rm -f $(ALL_OBJS) $(ALL_LIBS) $(ALL_BIN)

install-conf:
		install -m 0755 -d $(HOME)/meteor/etc
		install -m 0644 conf/trajectorysolution.conf $(HOME)/meteor/etc

install-base:
		install -m 0755 -d $(HOME)/meteor/lib
		install -m 0644 $(ALL_LIBS) $(HOME)/meteor/lib
		ln -sf $(LIBTS_SO) $(HOME)/meteor/lib/$(LIBTS_SONAME)

		install -m 0755 -d $(HOME)/meteor/include/trajectorysolution
		install -m 0644 $(ALL_INCL) $(HOME)/meteor/include/trajectorysolution

INSTALL=install-base

ifeq ($(strip $(have_conf)),)
INSTALL+=install-conf
endif

install:	$(INSTALL)