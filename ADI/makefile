CC = nvcc
CFLAGS = -O3
SOURCES = adi.cu
TARGET = adi

$(TARGET) : $(SOURCES)
	$(CC) -Xcompiler=$(CFLAGS) -o $@ $^
