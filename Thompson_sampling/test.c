#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <string.h>

int main() {
    const char* FILENAME = "ts_shared_memory";
    const int SIZE = 1024;

    // 打开共享内存文件
    int fd = open(FILENAME, O_RDWR);
    if (fd == -1) {
        perror("open");
        exit(1);
    }

    // 映射共享内存
    char* shared_memory = (char*)mmap(NULL, SIZE, PROT_READ, MAP_SHARED, fd, 0);
    if (shared_memory == MAP_FAILED) {
        perror("mmap");
        exit(1);
    }
    
    unsigned char value = 0x03;
    off_t offset = 0;
    ssize_t bytes_written = pwrite(fd, &value, sizeof(value), offset);
    if (bytes_written == -1) {
        perror("pwrite");
        exit(1);
    }
    
    offset = lseek(fd, 1, SEEK_SET);
    if (offset == -1) {
        perror("lseek");
        close(fd);
        return 1;
    }
    
    int a = 100;
    char index[20];
    sprintf(index, "%u", a);
    size_t index_len = strlen(index);
    bytes_written = write(fd, index, index_len);
    if (bytes_written == -1) {
        perror("write");
        exit(1);
    }

    // 获取互斥锁
    char* lock = shared_memory;

    // 阻塞等待互斥锁释放
    while (*lock != 0) {
        sleep(0.01);
    }

    // 输出共享数据
    printf("Data from Python: %s\n", shared_memory + 1);
    
    char clear[1024];
    for (int i = 0; i < 1024; ++i) {
        clear[i] = '\x00';
    }

    if (write(fd, clear, sizeof(clear)) == -1) {
        perror("write");
        close(fd);
        return 1;
    }
    
    value = 0x02;
    offset = 0;
    bytes_written = pwrite(fd, &value, sizeof(value), offset);
    if (bytes_written == -1) {
        perror("pwrite");
        exit(1);
    }
    
    offset = lseek(fd, 1, SEEK_SET);
    if (offset == -1) {
        perror("lseek");
        close(fd);
        return 1;
    }
    
    // 写入字符串
    char* data = "1";
    size_t data_len = strlen(data);
    bytes_written = write(fd, data, data_len);
    if (bytes_written == -1) {
        perror("write");
        exit(1);
    }
    
    offset = lseek(fd, strlen(data) + 1, SEEK_SET);
    if (offset == -1) {
        perror("lseek");
        close(fd);
        return 1;
    }
    
    const char *str = ",0";
    ssize_t str_len = strlen(str);
    ssize_t write_size = write(fd, str, str_len);
    if (write_size != str_len) {
        perror("write");
        close(fd);
        return 1;
    }

    // 解除映射并关闭文件
    if (munmap(shared_memory, SIZE) == -1) {
        perror("munmap");
        exit(1);
    }

    if (close(fd) == -1) {
        perror("close");
        exit(1);
    }

    return 0;
}
