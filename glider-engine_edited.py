#! /usr/bin/python

# flt-times.py program to extract the flight times from GPS trace data

import sys
from array import array
import os
import subprocess
import numpy as np
import math as ma
import matplotlib.pyplot as plt
# plt.style.use('seaborn-white')
from datetime import timedelta
from datetime import datetime
from geopy import distance
from itertools import islice
import pyproj as py
from osgeo import gdal, osr
import rasterio as rio

# function to generate speed value every 5 msec, plot that usin pyplot lib
def c_time(file, fnout, dband1):

  fn = open(file, encoding='utf-8', errors='ignore')
  igcfn = str(file).split("/")[-1]
  atime = alat = alon = aN = aW = aF = apress = agnss = btime = blen = 0
  spd = start = st = spress = bcnt = stop = hpress = mpress = dpress = 0
  eng = rpm = enl = mop = engval = enlval = mopval = engflg = enlflg = mopflg = 0
  fdate = 'Unknown'
  seng = []
  agleng = []
  msleng = []
  senl = []
  aglenl = []
  mslenl = []
  smop = []
  aglmop = []
  mslmop = []
  surface_height_set = False
  surface_height = pressure_altitude = alt_offset = None
  Iline = ''
  FMT = '%H%M%S'
  gid = 'Unknown'
  gtype = 'Unknown'
  cid = 'Unknown'
  pname = 'P.Pilot'
  HAT = onc = maxaglalt = aglalt = ilen = 0
  shat = []
  mhat = []
  that = []
  ahat = []
  S2H = 3600
  M2F = 3.28084
  K2M = .621371
  xtime = '00:03:30'
  tas = gsp = sginit = demalt = dist = 0

  # We'll accumulate engine/sensor print lines for the "Sensor Info" CSV column
  sensor_info = []

  try:
    print('Adding Geoheight to the GPS altitude')
    gpath = ('us_nga_egm08_25.tif')
    inRas = gdal.Open(gpath)
    if inRas is None:
      print ('Could not open image file')
      sys.exit(1)

    # read in the geoid raster
    gband1 = inRas.GetRasterBand(1)
    rows = inRas.RasterYSize
    cols = inRas.RasterXSize
    cropData = gband1.ReadAsArray(0,0,cols,rows)
    transform = inRas.GetGeoTransform()
    xOrigin = transform[0]
    yOrigin = transform[3]
    pixWidth = transform[1]
    pixHeight = transform[5]

    for line in fn:
        line = line.strip().rstrip("\n").rstrip("\r")
        if not line:
            continue
        igc = ''.join(islice(line, 1)) 

        if igc == 'H':
            hstr = line.split(":")
            if hstr[0] == 'HFGTYGLIDERTYPE':
                gtype = ''.join((hstr[1]).split())
                continue
            hdate = ''.join(islice(line, 5))
            if hdate == 'HFDTE' :
                res = str(line).split(":")
                if len(res) > 1 :
                    res2 = res[1].split(",")
                    if len(res2) > 1 :
                        fltdate = res2[0]
                        print(fltdate)
                    else :
                        fltdate = res
                else :
                    fltdate = (''.join(islice(line, 5, 12)))
                fday = (''.join(islice(fltdate, 0, 2)))
                fmon = (''.join(islice(fltdate, 2, 4)))
                fyr = (''.join(islice(fltdate, 4, 6)))
                fdate = (str(fmon) + '/' + str(fday) + '/20' + str(fyr))
                if (str(fdate) == '') :
                    fdate = 'Unknown'
                continue

        if igc ==  'I':
            cnt = ''.join(islice(line, 1, 3))
            if ((not cnt.isdigit()) or (int(cnt) == 0)):
                continue
            Iline = line
            j = 7
            k = len(line)
            for i in range(int(cnt)):
                tag = ''.join(islice(line, j, j+3))
                if tag == 'TAS':
                    tas = ''.join(islice(line, j-4, j-2))
                if tag == 'GSP':
                    gsp = ''.join(islice(line, j-4, j-2))
                if tag == 'RPM':
                    rpm = ''.join(islice(line, j-4, j-2))
                if tag == 'MOP':
                    mop = ''.join(islice(line, j-4, j-2))
                if tag == 'ENL':
                    enl = ''.join(islice(line, j-4, j-2))
                j = j+7

            if int(enl) > 0:
                eng = enl
                sensor = "ENL"
            if int(mop) > 0:
                eng = mop
                sensor = "MOP"
            if int(rpm) > 0:
                eng = rpm
                sensor = "RPM"

            ilen = ''.join(islice(line, k-5, k-3))
            continue

        if igc == 'B':
            if blen == 0:
                blen = 1
                if (ilen == 0):
                    ilen = len(line)
            if (len(line) != int(ilen)):
                continue

            bcnt = bcnt + 1
            btime = atime
            blat = alat
            bN = aN
            blon = alon
            bW = aW 
            bpress = apress
            bgnss = agnss

            atime = ''.join(islice(line, 1, 7))
            asec = ''.join(islice(atime, 4, 6))
            if (str(atime) == '000000'):
                atime = btime
                bcnt = bcnt - 1
                continue

            aN = ''.join(islice(line, 14, 15))
            aW = ''.join(islice(line, 23, 24))
            if not (((aN == 'N') or (aN == 'S')) and ((aW == 'E') or (aW == 'W'))):
                atime = btime
                aN = bN
                aW = bW 
                bcnt = bcnt - 1
                continue

            alata = ''.join(islice(line, 7, 9))
            if ((int(alata) == 0) or (int(alata) > 90)):
                atime = btime
                bcnt = bcnt - 1
                continue

            alatb = ''.join(islice(line, 9, 14))
            alatc = '{:0.5f}'.format(int(alatb) / 60000 )
            alatd = str(alatc).split('.')
            alat = str(alata) + '.' + (str(alatd[1]))

            alona = ''.join(islice(line, 15, 18))
            alonb = ''.join(islice(line, 18, 23))
            alonc = '{:0.5f}'.format(int(alonb) / 60000 )
            alond = str(alonc).split('.')
            alon = str(alona) + '.' + (str(alond[1]))

            aF = ''.join(islice(line, 24, 25))
            if aW == 'W':
                alon = '-' + str(alon)
            if aN == 'S':
                alat = '-' + str(alat)

            apnt = (float(alat), float(alon))
            bpnt = (float(blat), float(blon))
            if (float(blat) > 0):
                dist = distance.distance(apnt, bpnt).m

            apress = ''.join(islice(line, 25, 30))
            agnss = ''.join(islice(line, 30, 35))

            if ((aF == 'V') or (int(dist) > 5000) or (int(asec) == 60) or
                (int(apress) < -500) or (int(apress) == 0) or
                ((abs(int(bpress)-int(apress))) > 800 and (bcnt > 1))):
                atime = btime
                alat = blat
                aN = bN
                aW = bW 
                apress = bpress
                agnss = bgnss
                bcnt = bcnt - 1
                continue

            if not surface_height_set and bcnt == 1:
                dem = [demdata.index(float(alon), float(alat))]
                demx = dem[0][0]
                demy = dem[0][1]
                demalt = dband1[demx, demy]
                dpress = int(apress) - int(demalt)
                if (int(dpress) > 150):
                    dpress = 0

                            # Store these values for output
                surface_height = int(demalt)
                pressure_altitude = int(apress)
                alt_offset =  int(dpress)
                surface_height_set = True
                
                spnt = (float(alat), float(alon))
                
                print(f"Calculation point: Pressure Alt: {pressure_altitude} ft MSL; Surface: {surface_height} ft MSL; Offset: {alt_offset} ft MSL")

            apress = int(apress) - int(dpress)
            if int(apress) > int(mpress):
                mpress = apress

            if ((sginit == 0) and (int(apress) > 0)):
                if dpress == 0:
                    spress = demalt
                else:
                    spress = apress
                sginit = 1

            if btime == 0:
                continue

            dtime = datetime.strptime(atime, FMT) - datetime.strptime(btime, FMT)
            dsec = dtime.total_seconds()
            if dsec == 0:
                continue

            pspd = spd

            # parse engine fields
            if int(eng) >= 1:
                i = int(eng)-1
                engval = ''.join(islice(line, i, i+3))
            if int(mop) >= 1:
                i = int(mop)-1
                mopval = ''.join(islice(line, i, i+3))
            if int(enl) >= 1:
                i = int(enl)-1
                enlval = ''.join(islice(line, i, i+3))

            if int(gsp) >= 1:
                i = int(gsp)-1
                sp = ''.join(islice(line, i, i+3))
                spd = int(sp)  * K2M
            else:
                spd = (dist / dsec)  * M2F / 5280 * S2H

            if (spd > (15*pspd)):
                continue

            mslalt = M2F*float(apress) 
            dem = [demdata.index(float(alon), float(alat))]
            demx = dem[0][0]
            demy = dem[0][1]
            demalt = int(M2F*dband1[demx, demy])
            aglalt = mslalt - demalt
            if aglalt > maxaglalt:
                maxaglalt = aglalt

            # detect MOP engine run
            if ((start != 0) and (int(mopval) > 300) and (mopflg == 0)):
                mopflg = 1
                smop.append(atime)
                mslmop.append(mslalt)
                aglmop.append(int(aglalt))

            if ((mopflg == 1) and (int(mopval) < 50) and (int(mop) > 0)):
                mopflg = 0
                smop.append(atime)
                mslmop.append(mslalt)
                aglmop.append(int(aglalt))

            # detect engine start
            if ((start != 0) and (int(engval) > 0) and (engflg == 0)):
                if (((sensor == "MOP") and (int(engval) > 600)) or
                    ((sensor == "ENL") and (int(engval) > 700)) or
                    ((sensor == "RPM") and (int(engval) > 150))):
                    engflg = 1
                    if (int(btime) == int(start)):
                        seng.append(start)
                        msleng.append(M2F*float(bpress))
                        agleng.append((M2F*float(bpress) - demalt))
                    else:
                        seng.append(atime)
                        msleng.append(mslalt)
                        agleng.append(aglalt)

            # detect engine stop
            if (engflg == 1):
                if (((sensor == "MOP") and (int(engval) < 50)) or
                    ((sensor == "ENL") and (int(engval) < 250)) or
                    ((sensor == "RPM") and (int(engval) < 20))):
                    engflg = 0
                    seng.append(atime)
                    msleng.append(mslalt)
                    agleng.append(aglalt)

            # detect ENL-based engine start/stop
            if ((start != 0) and (int(enlval) > 600) and (enlflg == 0) and (int(rpm) == 0)):
                enlflg = 1
                senl.append(atime)
                mslenl.append(mslalt)
                aglenl.append(int(aglalt))

            if ((enlflg == 1) and (int(enlval) < 250) and (rpm == 0)):
                enlflg = 0
                senl.append(atime)
                mslenl.append(mslalt)
                aglenl.append(int(aglalt))

            # detect start of flight
            if spd >= 35:
                if start == 0:
                    start = atime
                    hpress = int(spress) + 180

        # AFTER reading a B-record, check landing logic
        if btime == 0:
            continue

        # Landing detection
        if ((spd <= 15) and (aglalt <= 200) and (start != 0)):
            st += 1
            if st <= 5:
                continue
            stop = atime
            ldist = distance.distance(spnt, bpnt).m
            if ldist > 1500:
                lo = "LOUT"
            else:
                lo = "HOME"

            fltime = datetime.strptime(stop, FMT) - datetime.strptime(start, FMT)
            flc = ''.join(islice(str(fltime), 1))
            if flc == '-':
                res = str(fltime).split(",")
                str(res[1]).split()
                ftime = res[1]
            else:
                ftime = fltime
            start_alt= str(int(M2F*int(spress))) 
            # EXACT original prints:
            print('Glider: ' + str(gtype) + ' Date: ' + str(fdate) + ' Flight Time: ' + str(ftime) + ' Landing: ' + str(lo) + ' Start Alt: ' + start_alt + ' ft MSL')
            print('Start Time: ' + str(start) + ' Stop Time: ' + str(stop) + ' Max Altitude: ' + str(int(M2F*int(mpress))) + '[' + str(int(maxaglalt)) + '] ft MSL/ft AGL')

            # We'll store the "Max Altitude" from that line in a variable for CSV
            max_alt_str = f"{int(M2F*int(mpress))}[{int(maxaglalt)}]"

            # Engine runs
            i = 0
            while (i < len(seng)):
                s = seng[i]
                if (i+1 == len(seng)):
                    ss = atime
                    hgain = int(mslalt) - int(msleng[i])
                else:
                    ss = seng[i+1]
                    hgain = int(msleng[i+1]) - int(msleng[i])
                rntime = datetime.strptime(ss, FMT) - datetime.strptime(s, FMT)
                runtime = round(rntime.total_seconds())/60
                if (ma.floor(int(runtime)) > 0):
                    line_to_print = (
                        gtype + "'s " + sensor +
                        " monitor reports Engine Run " + str(int(runtime)) +
                        " minutes, starts at T=" + str(int(s)) +
                        " and: " + str(int(msleng[i])) + " msl [" +
                        str(int(agleng[i])) + " agl]; Height gain/loss is: " + str(int(hgain))
                    )
                    print(line_to_print)
                    # also store for CSV
                    sensor_info.append(line_to_print)
                i += 2

            # ENL sensor
            if (len(senl) > 0):
                line_to_print = (gtype + " Motor noise registered by ENL sensor at t=" +
                                 str(senl) + " and " + str(aglenl) + "AGL")
                print(line_to_print)
                sensor_info.append(line_to_print)

            # MOP sensor
            if (len(smop) > 0):
                line_to_print = (gtype + " Motor noise registered by MOP sensor at t=" +
                                 str(smop) + " and " + str(aglmop) + "AGL")
                print(line_to_print)
                sensor_info.append(line_to_print)

            print('\n\n')

            # Build the multiline Sensor Info string
            sensor_str = "\n".join(sensor_info).replace('"','""')  # escape quotes for CSV

            # Write to CSV with new columns:
            #   Landing, Max Altitude, Sensor Info
            #print("Debug: sensor string::",sensor_str)

            fnout.write(
          f"{fdate},{igcfn},{gtype},{ftime},{start},{stop},{lo},{start_alt},{max_alt_str},{surface_height},{pressure_altitude},{alt_offset},\"{sensor_str}\"\n"
    )

            # Reset flight vars
            HAT = onc = 0
            shat = []
            mhat = []
            that = []
            ahat = []
            atime = alat = alon = aN = aW = aF = apress = agnss = btime = 0
            spd = start = st = spress = bcnt = stop = hpress = mpress = sginit = 0
            eng = rpm = enl = mop = engval = enlval = mopval = engflg = enlflg = mopflg = 0
            seng = []
            msleng = []
            agleng = []
            senl = []
            smop = []
            aglmop = []
            mslmop = []
            maxaglalt = aglalt = 0
            sensor_info = []

  except Exception as e:
      print(e)
      print('Exception occurred, go to next file')
      return

  # If no stop found but we had a start
  if ((stop == 0) and (start != 0)):
      print('End) of Trace, No stop time found, print anyway')
      stop = atime
      if onc == 2:
          etime = atime
          ltime = datetime.strptime(atime, FMT) - datetime.strptime(etime, FMT)
          lsec = ltime.total_seconds()
          shat.append(lsec)
          mhat.append(xpress)

      fltime = datetime.strptime(stop, FMT) - datetime.strptime(start, FMT)
      ldist = distance.distance(spnt, bpnt).m
      if ldist > 1500:
          lo = "LOUT"
      else:
          lo = "HOME"
      flc = ''.join(islice(str(fltime), 1))
      if flc == '-':
          res = str(fltime).split(",")
          str(res[1]).split()
          ftime = res[1]
      else:
          ftime = fltime

      print('Glider: ' + str(gid) + ' Date: ' + str(fdate) + ' Flight Time: ' + str(ftime) + ' Landing: ' + str(lo) + ' Start Alt: ' + start_alt + ' ft MSL')
      print('Start Time: ' + str(start) + ' Stop Time: ' + str(stop) + ' Max Altitude: ' + str(int(M2F*int(mpress))) + '[' + str(int(maxaglalt)) + '] ft MSL/ft AGL')
      
      max_alt_str = f"{int(M2F*int(mpress))}[{int(maxaglalt)}]"

      i = 0
      while (i < len(seng)):
          s = seng[i]
          if (i+1 == len(seng)):
              ss = atime
              hgain = int(mslalt) - int(msleng[i])
          else:
              ss = seng[i+1]
              hgain = int(msleng[i+1]) - int(msleng[i])
          rntime = datetime.strptime(ss, FMT) - datetime.strptime(s, FMT)
          runtime = round(rntime.total_seconds())/60
          if (ma.floor(int(runtime)) > 0):
              line_to_print = (
                  sensor + " monitor reports Engine Run " + str(int(runtime)) +
                  " minutes, starts at T=" + str(int(s)) +
                  " and: " + str(int(msleng[i])) + " msl [" +
                  str(int(agleng[i])) + " agl]; Height gain/loss is: " + str(int(hgain))
              )
              print(line_to_print)
              sensor_info.append(line_to_print)
          i += 2

      if (len(senl) > 0):
          line_to_print = (gtype + " Motor noise registered by ENL sensor at t=" + str(senl) + " and " + str(aglenl) + "AGL")
          print(line_to_print)
          sensor_info.append(line_to_print)

      if (len(smop) > 0):
          line_to_print = (gtype + " Motor noise registered by MOP sensor at t=" + str(smop) + " and " + str(aglmop) + "AGL")
          print(line_to_print)
          sensor_info.append(line_to_print)

      sensor_str = "\n".join(sensor_info).replace('"','""')

      
      fnout.write(
          f"{fdate},{igcfn},{gtype},{ftime},{start},{stop},{lo},{start_alt},{max_alt_str},{surface_height},{pressure_altitude},{alt_offset},\"{sensor_str}\"\n"
    )


if (len(sys.argv) == 1):
    print("Usage: flt-presalt.py [dir]/ \n")
    sys.exit(0)

# We update the CSV header to include 2 new columns: "Max Altitude" and "Sensor Info".
fnout = open("Flt-times." + str(os.getpid()) + ".csv", "w")
fnout.write("Date (MM/DD/YYYY),File,Gtype,Flight Time,Start Time,End Time,Landing,Start_Alt (ft MSL),Max Altitude (ft MSL/ft AGL),Surface Height (ft MSL),Pressure Altitude (ft MSL),Offset (ft),Sensor Info\n")

# Updated fnout.write to include surface height, pressure altitude, and offset

print("Adding DEM heights for each lat/long")
demdata = rio.open('conus.tif')
dband1 = demdata.read(1)

import glob
files = glob.glob(f"{sys.argv[1]}/*.IGC") + glob.glob(f"{sys.argv[1]}/*.igc")
if not files:
    print("No .IGC or .igc files found")
    sys.exit(1)

for file in files:
    try:
        c_time(file, fnout, dband1)
    except Exception as e:
        print(f"Error processing file {file}: {e}")

fnout.close()
