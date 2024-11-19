CC = nvcc
CFLAGS = -O3
SOURCES = jacoby.cu
TARGET = jacoby

$(TARGET) : $(SOURCES)
	$(CC) -Xcompiler=$(CFLAGS) -o $@ $^
