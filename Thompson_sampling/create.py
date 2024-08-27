import mmap

# 创建共享内存文件
fd = open("ts_shared_memory", "w+b")
fd.write(b"\x01" + b"\x00" * (10000000 - 1))  # 写入一个字节的锁和1023字节的数据
fd.close()
