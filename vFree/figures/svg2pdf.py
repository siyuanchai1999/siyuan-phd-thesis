import os
import sys
from subprocess import Popen

def svg2pdf(src, dst):
    x = Popen([sys.argv[1], src, \
        '--export-filename=%s' % dst])
    try:
        waitForResponse(x)
    except:
        return False

def waitForResponse(x): 
    out, err = x.communicate() 
    if x.returncode < 0: 
        r = "Popen returncode: " + str(x.returncode) 
        raise OSError(r)
    
files = [f for f in os.listdir('.') if os.path.isfile(f)]
if len(sys.argv) > 2:
    files = sys.argv[2:]

for src in files:
    if not src.endswith('.svg'):
        continue
    dst = src.replace('.svg', '.pdf')
    print('Rendering: ', src, ' -> ', dst)
    svg2pdf(src, dst)
