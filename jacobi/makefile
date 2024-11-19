CC = nvcc
CFLAGS = -O3
SOURCES = jacoby.cu
OBJECTS = $(SOURCES:.c=.o)
TARGET = jacoby

$(TARGET) : $(OBJECTS)
	$(CC) -Xcompiler=$(CFLAGS) -o $@ $^
