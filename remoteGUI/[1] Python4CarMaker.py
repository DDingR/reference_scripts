# Importieren der Python Bibliotheken (= #include in C)
import socket
import time
import sys, string, os

# Setzen von Variablen
#    Pfad zu CM
#CM_PATH = "C:\IPG\hil\win32-5.1.3\bin\CM.exe"
CM_PATH = "/opt/ipg/carmaker/linux64/bin/CM"
#    Rechner auf dem der TCP/IP port geoeffnet wird
TCP_IP = 'localhost'
#    Portnummer
TCP_PORT = 16660 
BUFFER_SIZE = 1024

# "os.system" startet ein Programm
# Hier: Start von CM.exe mit der Option -cmdport 
#	(--> TCP/IP Port auf CM Seite oeffnen)
os.system("%s -cmdport %s &" % (CM_PATH,TCP_PORT)) 

# Warten bis CM GUI gestartet
time.sleep(2)

# Oeffnen des TCP/IP Ports in Python
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# Verbinden mit dem CM TCP/IP Port 16660 auf localhost
s.connect((TCP_IP, TCP_PORT))

# Senden von ScriptControl-Befehlen ueber den Port
# Mehere Befehle pro Nachricht moeglich; nach jedem Befehl \r
#MESSAGE = "LoadTestRun Examples/VehicleDynamics/Braking\rQuantSubscribe {Time}\rStartSim\r"

MESSAGE = "LoadTestRun testrun2\rQuantSubscribe {Time}\rStartSim\r"
MESSAGE = MESSAGE.encode('utf-8')
s.send(MESSAGE)
time.sleep(500)

# Senden eines ScriptControl-Befehls mit Rueckgabewert 
# und Empfangen dieses Rueckgabewertes 
# ("0" wird automatisch davor gesetzt --> nicht unterdrueckbar?)
MESSAGE = "expr {$Qu(Time)*1000}\rStopSim\r"
MESSAGE = MESSAGE.encode('utf-8')
s.send(MESSAGE)
time.sleep(1)
s_string_val = s.recv(200)
# Ausgabe des empfangenen Wertes in Python
print (s_string_val)

# Schliessen des Ports
s.close()
