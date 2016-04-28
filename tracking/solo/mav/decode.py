import struct
import numpy as np
import datetime
import os,sys

TIME_LEN=8
HEADER_LEN=5




def convertLog(logName):
    path, filename = os.path.split(logName)
    noext, ext = os.path.splitext(filename)
    
    outfilename = path + '/' + noext + '.csv'
    
    outfile = open(outfilename, "w")

    
    f = open(logName, "rb")

    bytesize = f.seek(0,2)
    f.seek(0,0)
    
    # convert the bytelist to ints to check for msg headers
    bytelist = np.zeros(bytesize,dtype='uint8')
    for i in range(bytesize):
        bt = f.read(1)
        bytelist[i]=struct.unpack('1B',bt)[0]
    
    
    first = True
    for i in range(TIME_LEN,bytesize-HEADER_LEN):
        if bytelist[i]==254:
            if bytelist[i+1]==28 and bytelist[i+5]==33:
                #time_boot_ms	uint32_t	Timestamp (milliseconds since system boot)
                #lat	int32_t	Latitude, expressed as * 1E7
                #lon	int32_t	Longitude, expressed as * 1E7
                #alt	int32_t	Altitude in meters, expressed as * 1000 (millimeters), AMSL (not WGS84 - note that virtually all GPS modules provide the AMSL as well)
                #relative_alt	int32_t	Altitude above ground in meters, expressed as * 1000 (millimeters)
                #vx	int16_t	Ground X Speed (Latitude), expressed as m/s * 100
                #vy	int16_t	Ground Y Speed (Longitude), expressed as m/s * 100
                #vz	int16_t	Ground Z Speed (Altitude), expressed as m/s * 100
                #hdg	uint16_t	Compass heading in degrees * 100, 0.0..359.99 degrees. If unknown, set to: UINT16_MAX
                f.seek(i-8,0)
                time_micro=struct.unpack('>1Q',f.read(8))[0]
                time_stamp = datetime.datetime.fromtimestamp(time_micro/1000000)
                outfile.write(time_stamp.strftime("%d-%m-%Y:%H:%M:%S.%f"))
                if first:
                    print(time_stamp.strftime("%d-%m-%Y:%H:%M:%S.%f"))
                    first=False
                outfile.write(",GLOBAL_POSITION_INT")

                f.seek(i+6,0)
                time_boot=struct.unpack('<1I',f.read(4))[0]
                outfile.write(",time_boot_ms,{}".format(time_boot))
                lat=struct.unpack('<1i',f.read(4))[0]
                outfile.write(",lat,{}".format(lat))
                lon=struct.unpack('<1i',f.read(4))[0]
                outfile.write(",lon,{}".format(lon))
                alt=struct.unpack('<1i',f.read(4))[0]
                outfile.write(",alt,{}".format(alt))
                rel_alt=struct.unpack('<1i',f.read(4))[0]
                outfile.write(",relative_alt,{}".format(rel_alt))
                vx=struct.unpack('<1h',f.read(2))[0]
                outfile.write(",vx,{}".format(vx))
                vy=struct.unpack('<1h',f.read(2))[0]
                outfile.write(",vy,{}".format(vy))
                vz=struct.unpack('<1h',f.read(2))[0]
                outfile.write(",vz,{}".format(vz))
                hdg=struct.unpack('<1H',f.read(2))[0]
                outfile.write(",hdg,{}".format(hdg))
                

                outfile.write("\n")
            if bytelist[i+1]==1 and bytelist[i+5]==215:
                #time_boot_ms	uint32_t	Timestamp (milliseconds since system boot)
                f.seek(i-8,0)
                time_micro=struct.unpack('>1Q',f.read(8))[0]
                try:
                    time_stamp = datetime.datetime.fromtimestamp(time_micro/1000000)
                except ValueError:
                    continue
                outfile.write(time_stamp.strftime("%d-%m-%Y:%H:%M:%S.%f"))
                if first:
                    print(time_stamp.strftime("%d-%m-%Y:%H:%M:%S.%f"))
                    first=False
                outfile.write(",GOPRO_HEARTBEAT")

                f.seek(i+6,0)
                record_value=struct.unpack('<1B',f.read(1))[0]
                outfile.write(",record_value,{}".format(record_value))
                

                outfile.write("\n")
               
               
                    
    outfile.close()

if __name__ == '__main__':
    
    
    
    if len(sys.argv)>1:
        convertLog(sys.argv[1])
    else:
        convertLog("log_udp_2015_10_26_11_36_19.tlog")        
    

