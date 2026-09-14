"""Resumable range downloads from the official OGBench source, with pinned checksums."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import zipfile

ROOT = Path('/root/autodl-tmp/ogbench_data')
FILES = {
    'antmaze-large-navigate-v0.npz': (242877817, '2dbe946ac107ca5ac524f82f7a72efa951ceab0be915201d6b32e612d823f2e7'),
    'antmaze-large-navigate-v0-val.npz': (24302303, 'e9a4126825faefb57c34c29f1ee423858dbc4b6f71d9a9846a8b8bdb908b44a1'),
}
BLOCK = 1024*1024

def download(name, size, sha):
    dst = ROOT/name
    if dst.exists() and dst.stat().st_size == size and hashlib.sha256(dst.read_bytes()).hexdigest() == sha:
        return
    parts = ROOT/(name+'.parts')
    parts.mkdir(exist_ok=True)
    origin = os.environ.get('ANTMAZE_DOWNLOAD_ORIGIN',
        'https://huggingface.co/datasets/yonghoon96/ogbench-mirror/resolve/main')
    # Mirror LFS SHA256 values match complete local downloads from Berkeley.
    url = origin+'/'+name
    def fetch(i):
        start, end = i*BLOCK, min(size,(i+1)*BLOCK)-1
        p = parts/str(i)
        if p.exists() and p.stat().st_size == end-start+1:
            return
        for attempt in range(5):
            tmp, hdr = p.with_suffix('.tmp'), p.with_suffix('.headers')
            r = subprocess.run(['curl','--http1.1','-sSL','--fail','--max-time','120',
                '--range',f'{start}-{end}','--dump-header',str(hdr),'-o',str(tmp),url],
                capture_output=True,text=True)
            if r.returncode == 0 and tmp.stat().st_size == end-start+1 and re.search(
                    fr'content-range: bytes {start}-{end}/{size}',hdr.read_text(),re.I):
                tmp.replace(p)
                print(name,'part',i,'complete',flush=True)
                return
            print(name,'retry',i,attempt,'code',r.returncode,flush=True)
        raise RuntimeError(f'Failed official download part {name}:{i}')
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(fetch, range((size+BLOCK-1)//BLOCK)))
    temp = dst.with_suffix('.verified-download')
    with temp.open('wb') as f:
        for i in range((size+BLOCK-1)//BLOCK):
            f.write((parts/str(i)).read_bytes())
    assert temp.stat().st_size == size
    assert hashlib.sha256(temp.read_bytes()).hexdigest() == sha
    with zipfile.ZipFile(temp) as z:
        assert z.testzip() is None
    temp.replace(dst)
    print(json.dumps(dict(file=name, bytes=size,sha256=sha,status='passed')),flush=True)

for name, spec in FILES.items():
    download(name, *spec)
(ROOT/'DOWNLOAD_VERIFIED.json').write_text(json.dumps(dict(status='passed',files=FILES),indent=2))
