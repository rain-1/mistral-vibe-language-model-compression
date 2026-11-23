import os, sys
EXTS = {'.c','.h','.S','.s','.txt','.rst','.dts','.dtsi','.yaml','.json','.pl','.py','.sh'}
NAMES = {'Makefile','Kconfig','Kbuild','COPYING','README','CREDITS','MAINTAINERS','TODO','REPORTING-BUGS','checkpatch.pl','get_maintainer.pl',' Lindent'}
def collect(root):
    out = []
    for dp, dn, fn in os.walk(root):
        dn[:] = [d for d in dn if d not in ('.git','tools','Documentation') or True]
        for f in fn:
            ext = os.path.splitext(f)[1].lower()
            if ext in EXTS or f in NAMES or f.startswith('Kconfig') or f.startswith('Makefile'):
                p = os.path.join(dp, f)
                try:
                    with open(p, 'r', errors='ignore') as fh:
                        out.append(fh.read())
                except Exception:
                    pass
    return '\n'.join(out)
for v in ['2.0.40','3.0','4.0','5.0','6.0']:
    root = f'/home/riv/kernels/linux-{v}'
    text = collect(root)
    with open(f'/home/riv/kernels/corpus_{v}.txt','w') as fh:
        fh.write(text)
    print(v, len(text), 'chars')
