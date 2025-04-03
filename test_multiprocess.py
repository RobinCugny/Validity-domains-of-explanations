import multiprocessing
from multiprocessing import Pool

def f(x):
  return x

with Pool(processes=multiprocessing.cpu_count()) as pool:
  for i in pool.imap(f, range(10)):
    print(i)