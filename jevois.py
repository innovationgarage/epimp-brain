import serial
from subprocess import Popen
from config import cfg
import argparse
import time
import Object as _
import io
import datetime
import numpy as np

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--model', dest='model', default='DarkNetYOLO', type=str)
parser.add_argument('--serial-in', dest='serialin', default='0', type=int)
parser.add_argument('--serial-out', dest='serialout', default='1', type=int)
args = parser.parse_args()

#f = open('serout.csv', 'w')
serin = serial.serial_for_url('/dev/ttyACM{}'.format(args.serialin), timeout=1)
sio = io.TextIOWrapper(io.BufferedRWPair(serin, serin))
sio.flush() # it is buffering. required to get the data out *now*
sio.write(unicode(cfg[args.model]['set_mapping']))
process = Popen(cfg[args.model]['start_video'])
time.sleep(5)
sio.write(unicode(cfg[args.model]['set_serout']))
time.sleep(1)
sio.write(unicode(cfg[args.model]['set_threshold']))
time.sleep(1)
if cfg[args.model]['set_serstyle']=='D2':
    sio.write(unicode('setpar serstyle Detail'))
elif cfg[args.model]['set_serstyle']=='T2':
    sio.write(unicode('setpar serstyle Terse'))
time.sleep(1)
recent_objects = []
t0 = time.time()
with open('serout.csv', 'w') as f:
    while serin.isOpen():
        line = sio.readline().split()
#        print(line)
        if line:
            if line:
                if line[0]=='D2':
                    found_object = _.D2(line)
                elif line[0]=='T2':
                    found_object = _.T2(line)
                properties = found_object.getProperties()
                for key, prop in properties.items():
                    f.write("%s," % prop)
                f.write('\n')
                t1 = properties['timestamp']
#                print(len(recent_objects), (t1-t0), 60./cfg[args.model]['fpm'])
                if((t1-t0)<60./cfg[args.model]['fpm']):
                    recent_objects.append(properties)
                else:
                    x = np.median([(el['x']) for el in recent_objects])
                    y = np.median([(el['y']) for el in recent_objects])
                    recent_objects = []
                    t0 = time.time()
                    print(int(time.time()), x, y)
