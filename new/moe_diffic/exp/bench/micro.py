"""GPU 微基准：矩阵乘峰值（fp32/tf32/bf16/fp16）、显存带宽、小内核启动开销。用法: python micro.py"""
import time, torch
d = 'cuda'
print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0), 'torch', torch.__version__)
def tflops(dt, tf32=False, n=8192, it=30):
    torch.backends.cuda.matmul.allow_tf32 = tf32
    a = torch.randn(n, n, device=d, dtype=dt); b = torch.randn(n, n, device=d, dtype=dt)
    for _ in range(5): a @ b
    torch.cuda.synchronize(); t = time.time()
    for _ in range(it): a @ b
    torch.cuda.synchronize(); return 2 * n ** 3 / ((time.time() - t) / it) / 1e12
print('matmul fp32 %.0f | tf32 %.0f | bf16 %.0f | fp16 %.0f TFLOPS' % (tflops(torch.float32), tflops(torch.float32, True), tflops(torch.bfloat16), tflops(torch.float16)))
x = torch.empty(1024 * 1024 * 512, device=d)
for _ in range(3): y = x * 2
torch.cuda.synchronize(); t = time.time()
for _ in range(20): y = x * 2
torch.cuda.synchronize(); print('显存带宽 %.0f GB/s' % (2 * x.numel() * 4 / ((time.time() - t) / 20) / 1e9))
s = torch.randn(256, device=d)
for _ in range(100): s = s * 1.0001
torch.cuda.synchronize(); t = time.time()
for _ in range(2000): s = s * 1.0001
torch.cuda.synchronize(); print('小内核启动开销 %.1f us/kernel' % ((time.time() - t) / 2000 * 1e6))
