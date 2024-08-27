import mmap
import os
import time

# 打开共享内存文件
with open("shared_memory", "r+b") as file:
    # 将文件映射为内存
    mm = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_WRITE)

    # 获取互斥锁
    lock = bytearray(mm[:1])

    # 写入数据
    data = "Hello from Python!"
    mm[1: len(data) + 1] = data.encode()

    # 释放互斥锁
    lock[0] = 0
    
    # 将修改后的互斥锁状态写回共享内存文件中
    mm.seek(0)
    mm.write(lock)
    
    while lock[0] != 2:
    	mm = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_WRITE)
    	lock = bytearray(mm[:1])
    	time.sleep(0.01)
    	
    print(mm[1:].decode())
    
    mm.write(b"\x01" + b"\x00" * 1023)

    # 解除内存映射
    mm.close()

