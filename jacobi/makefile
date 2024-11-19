CC = nvcc
CFLAGS = -O3
SOURCES = jacobi.cu
TARGET = jacobi

$(TARGET) : $(SOURCES)
	$(CC) -Xcompiler=$(CFLAGS) -o $@ $^
