import serial
def read_from_serial():
    ser = serial.Serial(
        port = '/dev/ttyACM0',
        baudrate = '115200',
        parity = serial.PARITY_NONE,
        stopbits = serial.STOPBITS_ONE,
        bytesize = serial.EIGHTBITS,
        timeout = 0
    )
    print("connected to: " + ser.portstr)
    ser.write("setpar serout All\n".encode())
    ser.write("setpar serstyle Detail\n".encode())
    # ser.write("setpar cfgfile cfg/yolo9000.cfg\n")
    ser.write("setpar cfgfile initscript.cfg\n".encode())
    # ser.write("setpar weightfile weights/yolo9000.weights\n")
    while True:
        line = ser.readline()
        if line:
            s = line
            print(s.split()),
        ser.close()

read_from_serial()        
