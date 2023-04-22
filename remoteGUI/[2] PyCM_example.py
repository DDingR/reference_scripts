# Import the Python libraries (= #include in C)
import socket
import time
import sys, string, os

# Setting variables
# Path to CM
CM_PATH = "C:\\IPG\\carmaker\\win64-11.1\\bin\\CM.exe"
Data_Directory = "D:\\CMProject\\CM11\\0829test"
# Computer on which the TCP / IP port is opened
TCP_IP = 'localhost'
# Portnumber
TCP_PORT = 16660
BUFFER_SIZE = 1024

# "os.system" starts a program
# Here: Start of CM.exe with the option -cmdport
# (-> open TCP / IP port on CM side)
os.system("%s %s -cmdport %s &" % (CM_PATH, Data_Directory, TCP_PORT))

# Wait for CM GUI to start
time.sleep(2)

# Open the TCP / IP port in Python
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# Connect to the CM TCP / IP port 16660 on localhost
s.connect((TCP_IP, TCP_PORT))

# Send ScriptControl commands over the port
# Several commands per message possible; after each command \r
MESSAGE = "LoadTestRun Examples/VehicleDynamics/Braking/Braking\rStartSim\r"
s.send(MESSAGE.encode())

MESSAGE = "WaitForStatus running\r"
s.sendall(MESSAGE.encode())

while 1:
  data = s.recv(BUFFER_SIZE)
  if '0' in data.decode():
      break

while 1:
  MESSAGE = "DVARead Car.v\r"
  s.send(MESSAGE.encode())
  vel = s.recv(BUFFER_SIZE)
  vel = (vel.decode()).strip()
  vel = float(vel[1:])
  print(vel)

  MESSAGE = "DVARead Time\r"
  s.send(MESSAGE.encode())
  t = s.recv(BUFFER_SIZE)
  t = (t.decode()).strip()
  t = float(t[1:])
  print(t)

  if vel > 10:
    MESSAGE = "DVAWrite DM.Brake 1 1000\r"
    s.send(MESSAGE.encode())
    s.recv(BUFFER_SIZE)

  if t > 30:
    MESSAGE = "StopSim\r"
    s.send(MESSAGE.encode())
    break


# Close the port
s.close()
