"""Download original OGBench files with bounded parallel HTTP ranges."""
import concurrent.futures
import hashlib
import os
from pathlib import Path
import time
import urllib.request

FILES = {
    'cube-double-play-v0.npz': (297435656, 'a73d1a33d029cedb8bc170ef94791ec585fa2d9450096f4f2a02b8cfbcf608c9'),
    'cube-double-play-v0-val.npz': (29726524, 'b1fcdf4bd40750351a58d0d491d6be198366ce898f0c6a2e4cb5db331966013e'),
}


def download(directory):
    directory=Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    for name,(size,expected) in FILES.items():
        dest=directory/name
        if dest.exists() and hashlib.sha256(dest.read_bytes()).hexdigest()==expected:
            continue
        partial=Path(str(dest)+'.parallel.part')
        fd=os.open(partial,os.O_RDWR|os.O_CREAT,0o600)
        os.ftruncate(fd,size)
        block=2*1024*1024
        ranges=[(start,min(start+block,size)-1) for start in range(0,size,block)]
        def get(pair):
            start,end=pair
            req=urllib.request.Request('http://rail.eecs.berkeley.edu/datasets/ogbench/'+name,
                                       headers={'Range':f'bytes={start}-{end}'})
            for attempt in range(4):
                try:
                    with urllib.request.urlopen(req,timeout=40) as response:
                        if response.status!=206 or response.headers['Content-Range']!=f'bytes {start}-{end}/{size}':
                            raise RuntimeError('Unexpected range response')
                        data=response.read()
                    assert len(data)==end-start+1
                    written=0
                    while written<len(data):
                        written+=os.pwrite(fd,data[written:],start+written)
                    return len(data)
                except Exception:
                    if attempt==3:
                        raise
                    time.sleep(1+attempt)
        try:
            total=0
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                for future in concurrent.futures.as_completed([executor.submit(get,r) for r in ranges]):
                    total+=future.result()
                    print(name,total,size,flush=True)
            os.fsync(fd)
        finally:
            os.close(fd)
        actual=hashlib.sha256(partial.read_bytes()).hexdigest()
        assert actual==expected,(name,actual,expected)
        partial.replace(dest)
        print(name,'SHA256 verified',actual,flush=True)


if __name__=='__main__':
    import sys
    download(sys.argv[1])
