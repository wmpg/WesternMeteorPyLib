# DetectionApp Makefile

CC=gcc

CFLAGS= -O2 -pipe -fPIC -W -Wextra -ggdb
CFLAGS+= -I$(HOME)/meteor/include

LDFLAGS= -Wl,-rpath,$(HOME)/meteor/lib
LDFLAGS+= -L$(HOME)/meteor/lib

OBJS=ParticleSwarmFunctions.o

default:	$(OBJS)

%.o:		%.c %.h
		$(CC) $(CFLAGS) -o $@ -c $<

clean:
		rm -f $(OBJS)
