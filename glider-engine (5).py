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
  # GEOD = py.Geod(ellps='WGS84')
    
  try :
    print('Adding Geoheight to the GPS altitude')
    gpath = ('us_nga_egm08_25.tif')
    inRas = gdal.Open(gpath)
    if inRas is None:
       print ('Could not open image file')
       sys.exit(1)
# 
    # read in the crop data and get info about it
    gband1 = inRas.GetRasterBand(1)
    rows = inRas.RasterYSize
    cols = inRas.RasterXSize
    cropData = gband1.ReadAsArray(0,0,cols,rows)
    transform = inRas.GetGeoTransform()
    xOrigin = transform[0] # x origin
    yOrigin = transform[3] # y origin
    pixWidth = transform[1] # pixel width. Assuming the source is in degrees, this is how many degrees per pixel
    pixHeight = transform[5] # pixel height
    # print("Adding DEM heights for each lat/long")
    # demdata = rio.open('/home/rcarlson/soaring/srtm/conus.tif')
    # dband1 = demdata.read(1)

# 
    for line in fn:
        # print(str(line))
        line = line.strip()
        line = line.rstrip("\n")
        line = line.rstrip("\r")
        # line = line.rstrip("\x00")
        igc = ''.join(islice(line, 1)) 
        # print('Character: ' + str(igc))
        if igc == 'H':
            hstr = line.split(":")
            if hstr[0] == 'HFGTYGLIDERTYPE':
                # gid = ''.join((hstr[1])[:-1].split())
                gtype = ''.join((hstr[1]).split())
                # print('HString: ' + str(hstr[0]) + ' GliderID: ' + str(gid) + '\n')
                continue
            hdate = ''.join(islice(line, 5))
            # print('Date: ' + hdate + '\n')
            if hdate == 'HFDTE' :
                res = str(line).split(":")
                if len(res) > 1 :
                    # print(res[1])
                    res2 = res[1].split(",")
                    # print(res2[0])
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
                # print(str(line))
                # print('Day/Month/year: ' + str(fday) + '/' + str(fmon) + '/' + str(fyr) + '\n')
                # fdate = (str(fyr) + '/' + str(fmon) + '/' + str(fday))
                # fdate = (str(fday) + '/' + str(fmon) + '/20' + str(fyr))
                fdate = (str(fmon) + '/' + str(fday) + '/20' + str(fyr))
                if (str(fdate) == '') :
                    fdate = 'Unknown'
                # print ('Date: ' + str(fdate) + '\n')
                continue
        if igc ==  'I':
            cnt = ''.join(islice(line, 1, 3))
            if ((not cnt.isdigit()) or (int(cnt) == 0)) :
                continue
            Iline = line
            j = 7
            k = len(line)
            for i in range(int(cnt)) :
                tag = ''.join(islice(line, j, j+3))
                # print('I Record tag is: ' + str(tag) + ' count(loop): ' + cnt + '(' + str(i) + ')')
                if tag == 'TAS' :
                    tas = ''.join(islice(line, j-4, j-2))
                    # print('TAS field starts at: ' + str(tas))
                if tag == 'GSP' :
                    gsp = ''.join(islice(line, j-4, j-2))
                    # print('GSP field starts at: ' + str(gsp))
                if tag == 'RPM' :
                    rpm = ''.join(islice(line, j-4, j-2))
                    # print('RPM field starts at: ' + str(rpm))
                if tag == 'MOP' :
                    mop = ''.join(islice(line, j-4, j-2))
                    # print('MOP field starts at: ' + str(mop))
                if tag == 'ENL' :
                    enl = ''.join(islice(line, j-4, j-2))
                    # print('ENL field starts at: ' + str(enl))
                j = j+7

            if int(enl) > 0 :
                eng = enl
                sensor = "ENL"
            if int(mop) > 0 :
                eng = mop
                sensor = "MOP"
            if int(rpm) > 0 :
                eng = rpm
                sensor = "RPM"
            ilen = ''.join(islice(line, k-5, k-3))
            continue
        if igc ==  'B':
            # print('length: ', len(line), ' : ', line)
            if (blen == 0) :
                # print('B record length: ', len(line), ' : ', str(ilen), ' : ', len(Iline), ' : ', Iline)
                blen = 1
                if (ilen == 0) :
                    ilen = len(line)
            if (len(line) != int(ilen)) :
                # print("Skipping short or corrupted B record")
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
            if (str(atime) == '000000') :
                # print('Reset btime to skip over day change\n')
                atime = btime
                bcnt = bcnt - 1
                continue
            aN = ''.join(islice(line, 14, 15))
            aW = ''.join(islice(line, 23, 24))
            if not (((aN == 'N') or (aN == 'S')) and ((aW == 'E') or (aW == 'W'))) :
                atime = btime
                aN = bN
                aW = bW 
                bcnt = bcnt - 1
                continue
            # print('ATime: ' + str(atime) + ' asec ' + str(asec))
            alata = ''.join(islice(line, 7, 9))
            if ((int(alata) == 0) or (int(alata) > 90)) :
                atime = btime
                bcnt = bcnt - 1
                continue
            alatb = ''.join(islice(line, 9, 14))
            alatc = '{:0.5f}'.format(int(alatb) / 60000 )
            alatd = str(alatc).split('.')
            # aN = ''.join(islice(line, 14, 15))
            # print('lata: ' + str(alata) + ' latb: ' + str(alatb) + ' latc: ' + str(alatc) + ' latd: ' + alatd[1] + '\n')
            # print('latDD: ' + str(alata) + ' latmm: ' + str(alatb) + ' N/S: ' + str(aN) + '\n')
            alat = str(alata) + '.' + (str(alatd[1]))
            alona = ''.join(islice(line, 15, 18))
            alonb = ''.join(islice(line, 18, 23)) 
            alonc = '{:0.5f}'.format(int(alonb) / 60000 )
            alond = str(alonc).split('.')
            # print('lonDD: ' + str(alona) + ' lonmm: ' + str(alonc) + ' E/W: ' + str(aW) + '\n')
            # print('lona: ' + str(alona) + ' lonb: ' + str(alonb) + ' lonc: ' + str(alonc) + ' lond: ' + alond[1] + '\n')
            alon = str(alona) + '.' + (str(alond[1]))
            # print('lat: ' + str(alat) + ' Long: ' + str(alon) + '\n')
            # aW = ''.join(islice(line, 23, 24))
            aF = ''.join(islice(line, 24, 25))
            if aW == 'W' :
                alon = '-' + str(alon)
            if aN == 'S' :
                alat = '-' + str(alat)
            # if ((aF == 'V') or (apress == '00000')) :
            apnt = (float(alat), float(alon))
            bpnt = (float(blat), float(blon))
            # xdist = distance.distance(aapnt, bbpnt).miles
            if (float(blat) > 0) :
                dist = distance.distance(apnt, bpnt).m
            # print('PointA: ' + str(apnt) + ' PointB: ' + str(bpnt) + ' Distance: ' + str(dist))
            apress = ''.join(islice(line, 25, 30))
            agnss = ''.join(islice(line, 30, 35))
            # print('aTime: ' + str(atime) + ' apress: ' + str(apress) + ' bpress: ' + str(bpress) + ' aflag: ' + str(aF))
            # if ((aF == 'V') or (int(apress) < -500) or ((int(apress) == 0) or ((int(apress)*30 < int(bpress)) and (int(apress) > 0)))) :
            if ((aF == 'V') or (int(dist) > 5000) or (int(asec) == 60)  or (int(apress) < -500) or (int(apress) == 0) or ((abs(int(bpress)-int(apress))) > 800) and (bcnt > 1)) :
                # print("Skipping to next B record at: " + str(atime) + ' [' + str(apress) + ']' + " bcnt: " + str(bcnt))
                atime = btime
                alat = blat
                aN = bN
                alon = blon
                aW = bW 
                apress = bpress
                agnss = bgnss
                bcnt = bcnt - 1
                continue
            # if aW == 'W' :
                # alon = '-' + str(alon)
            # if aN == 'S' :
                # alat = '-' + str(alat)
            # print('lat: ' + str(alat) + ' Long: ' + str(alon) + '\n')
            if (bcnt == 1) :
                # print('lat: ' + str(alat) + ' Long: ' + str(alon))
                # print(line)
                dem = [demdata.index(float(alon), float(alat))]
                demx = dem[0][0] 
                demy = dem[0][1]
                # print('Transformed lat/long ' + str(int(demx)) + '/' + str(int(demy)))
                demalt = dband1[demx, demy]
                dpress = int(apress) - int(demalt)
                if (int(dpress) > 150) :
                    dpress = 0
                print('Calculating Surface height and Alt offset at start Pressure Alt: ' + str(int(apress)) + '; Surface: ' + str(int(demalt)) + ' Offset: ' + str(int(dpress)))
                spnt = (float(alat), float(alon))
            apress = int(apress) - int(dpress)
            # print('New apress: ' + str(apress))
            if int(apress) > int(mpress) :
                mpress = apress
            # if ((start != 0) and (onc == 0)) :
                # print('Current alt: ' + str(apress) + ' Hard Alt: ' + str(int(spress)+180) + ' Prev alt: ' + str(bpress))
                # if int(apress) > int(hpress) :
                    # onc = 1
            # if onc == 1 :
                # if int(apress) < int(hpress) :
                    # HAT = HAT + 1
                    # print("glider dropped below Hard Deck Altitude (" + str(hpress) + ")  at: " + str(atime) + ", now at: " + str(apress) + " count: " + str(HAT))
                    # that.append(atime)
                    # ahat.append(apress)
                    # print('Hat alt list: ', ahat)
                    # print('Hat time list count: ', len(that), ' elements: ', that)
                    # xpress = apress
                    # etime = atime
                    # onc = 2
            # if onc == 2 :
                # if (int(apress) < int(xpress)) :
                    # print('Glider still below HD, now at: ' + str(apress))
                    # xpress = apress
                    # xtime = atime
                # if (int(apress) > int(hpress)) :
                    # print('Glider back above HD at: ' + str(atime) + ' began at: ' + str(etime))
                    # onc = 0
                    # ltime = datetime.strptime(atime, FMT) - datetime.strptime(etime, FMT)
                    # lsec = ltime.total_seconds()
                    # print('Glider was below HD for: ' + str(lsec)+ ' seconds, Lowest Alt: ' + str(xpress) + ' at: ' + str(xtime))
                    # shat.append(lsec)
                    # mhat.append(xpress)
                    # print('Hat Sec list count: ', len(shat), ' elements: ', shat)
                    # print('Hat MinAlt list count: ', len(mhat), ' elements: ', mhat)
            if ((sginit == 0) and (int(apress) > 0)) :
                if dpress == 0 :
                    spress = demalt
                else :
                    spress = apress
                sginit = 1
                # print("Initialize Start GNSS value to: ", str(spress))
            # print('apress: ' + str(apress) + ' bpress: ' + str(bpress) )
            paltd = int(apress) - int(bpress)
            # galtd = int(agnss) - int(bgnss)
            # print('apress: ' + str(apress) + ' bpress: ' + str(bpress) + ' paltd: ' + str(paltd))
            # print('agnss: ' + str(agnss) + ' bgnss: ' + str(bgnss) + ' galtd: ' + str(galtd) + " A-Alt: " + str(3.28084*int(agnss)))
            # if aW == 'W' :
                # alon = '-' + str(alon)
            # if aN == 'S' :
                # alat = '-' + str(alat)
            # print('lat: ' + str(alat) + ' Long: ' + str(alon) + '\n')
            if btime == 0 :
                continue
            dtime = datetime.strptime(atime, FMT) - datetime.strptime(btime, FMT)
            dsec = dtime.total_seconds()
            if dsec == 0 :
                continue
            # apnt = (alat, alon, int(apress))
            # bpnt = (blat, blon, int(bpress))
            # apnt = (alat, alon)
            # bpnt = (blat, blon)
            # apnt = (float(alat), float(alon))
            # bpnt = (float(blat), float(blon))
            # print('PointA: ' + str(apnt) + ' PointB: ' + str(bpnt))
            # xdist = distance.distance(aapnt, bbpnt).miles
            # dist = distance.distance(apnt, bpnt).m
            # xspd = (dist / dsec) * 3.28084 / 5280 * 3600 
            # if int(tas) >= 1 :
                # i = int(tas)-1
                # sp = ''.join(islice(line, i, i+3))
                # spd = int(sp) * K2M
            # elif int(gsp) >= 1 :
            pspd = spd
            if int(eng) >= 1 :
                i = int(eng)-1
                engval = ''.join(islice(line, i, i+3))
                # if (int(rpmval) > 0) :
                    # print('RPM value = ' + str(rpmval) + ' at T=' + str(atime))
            if int(mop) >= 1 :
                i = int(mop)-1
                mopval = ''.join(islice(line, i, i+3))
                # if (int(mopval) > 20) :
                    # print('MOP value = ' + str(mopval) + ' at T=' + str(atime))
            if int(enl) >= 1 :
                i = int(enl)-1
                enlval = ''.join(islice(line, i, i+3))
                # if ((int(enlval) >= 50) and (int(enlval) != int(enlx))) :
                # if int(enlval) >= 300 :
                    # print('ENL value = ' + str(enlval) + ' at T=' + str(atime))
            if int(gsp) >= 1 :
                i = int(gsp)-1
                sp = ''.join(islice(line, i, i+3))
                spd = int(sp)  * K2M
            else :
                spd = (dist / dsec)  * M2F / 5280 * S2H
            # print('distance: ' + str(dist) + ' seconds: ' + str(dsec) + ' speed: ' + str(spd) + ' [' + str(pspd) + '] Time: ' + str(atime) + ' Altitude: ' + str(apress) + '\n')
            if (spd > (15*pspd)) :
                    continue
            # xg = (float(alat) - yOrigin) / pixHeight
            # yg = (float(alon) - xOrigin) / pixWidth
            # print('geo coordinates ' + str(float(alon)) + ', ' + str(float(alat)))
            # geoalt = cropData[int(xg), int(yg)]
            # mslalt = M2F*(float(apress) - float(geoalt))
            mslalt = M2F*float(apress) 
            dem = [demdata.index(float(alon), float(alat))]
            demx = dem[0][0] 
            demy = dem[0][1]
            # print('Transformed lat/long ' + str(int(demx)) + '/' + str(int(demy)))
            demalt = int(M2F*dband1[demx, demy])
            aglalt = mslalt - demalt
            # if (((atime == '192015') or (atime == '192044') or (atime == '192140')) and (gid == 'Unknown')) :
                # print(str(atime) + ' [' + str(alat) + ', ' + str(alon) + '] Gnd[Glider: ' + str(gid) + '] is: ' + str(int(demalt)) + ' [' + str(int(mslalt) - (int(demalt))) + '] ft AGL; at: ' + str(int(spd)) + ' MPH')
            # if ((int(mslalt) - int(demalt)) <= 600) and (spd > 10) :
                # print(line)
                # print(str(atime) + ' [' + str(alat) + ', ' + str(alon) + '] [Glider: ' + str(gid) + '] Gnd is: ' + str(int(demalt)) + ' [' + str(int(mslalt) - (int(demalt))) + '] ft AGL; at: ' + str(int(spd)) + ' MPH')
                # print('Geoid Height at [' + str(alat) + ', ' + str(alon) + '] is: ' + str(int(geoalt)) + ' [' + str(int(mslalt)) + '] ft MSL')
            if aglalt > maxaglalt :
                maxaglalt = aglalt
            if ((start != 0) and (int(mopval) > 300) and (mopflg == 0)) :
                # print("Found Engine start based on RPM T=" + str(int(atime)) + " AGL = " + str(int(aglalt)))
                mopflg = 1
                smop.append(atime)
                mslmop.append(mslalt)
                aglmop.append(int(aglalt))
            if ((mopflg == 1) and (int(mopval) < 50) and (int(mop) > 0)) :
                # print("Found Engine stop based on RPM T=" + str(int(atime)) + " AGL = " + str(int(aglalt)))
                mopflg = 0
                smop.append(atime)
                mslmop.append(mslalt)
                aglmop.append(int(aglalt))
            if ((start != 0) and (int(engval) > 0) and (engflg == 0)) :
                # print("Found Engine start based on RPM T=" + str(int(atime)) + " AGL = " + str(int(aglalt)))
                if (((sensor == "MOP") and (int(engval) > 500)) or ((sensor == "ENL") and (int(engval) > 600)) or ((sensor == "RPM") and (int(engval) > 50))) :
                    engflg = 1
                    if (int(btime) == int(start)) :
                        seng.append(start)
                        msleng.append(M2F*float(bpress))
                        agleng.append((M2F*float(bpress) - demalt))
                    else :
                        seng.append(atime)
                        msleng.append(mslalt)
                        agleng.append(aglalt)
            # if ((engflg == 1) and (int(engval) == 0) and (int(eng) > 0)) :
            if (engflg == 1) :
                # print("Found Engine stop based on RPM T=" + str(int(atime)) + " AGL = " + str(int(aglalt)))
                if (((sensor == "MOP") and (int(engval) < 50)) or ((sensor == "ENL") and (int(engval) < 250)) or ((sensor == "RPM") and (int(engval) < 20))) :
                    engflg = 0
                    seng.append(atime)
                    msleng.append(mslalt)
                    agleng.append(aglalt)
            if ((start != 0) and (int(enlval) > 600) and (enlflg == 0) and (int(rpm) == 0)) :
                # print("Found Engine start based on Engine Noise T=" + str(int(atime)) + " AGL = " + str(int(aglalt)))
                enlflg = 1
                senl.append(atime)
                mslenl.append(mslalt)
                aglenl.append(int(aglalt))
            if ((enlflg == 1) and (int(enlval) < 250) and (rpm == 0)) :
                # print("Found Engine stop based on Engine Noise T=" + str(int(atime)) + " AGL = " + str(int(aglalt)))
                enlflg = 0
                senl.append(atime)
                mslenl.append(mslalt)
                aglenl.append(int(aglalt))
            if spd >= 35 :
                if start == 0 :
                     start = atime
                     hpress = int(spress) + 180
                     # print('New Start time: ' + str(start))
                # print('distance: ' + str(dist) + ' speed: ' + str(spd) + '\n')
        if btime == 0 :
            continue
        # print('speed: ' + str(spd) + ' aglalt: ' + str(aglalt) + " Time: " + str(atime) + ' Start: ' + str(start))
        # if ((spd <= 15) and (paltd == 0) and (start != 0)) :
        if ((spd <= 15) and (aglalt <= 200) and (start != 0)) :
            # print('speed: ' + str(spd) + ' aglalt: ' + str(int(aglalt)) + ' Start: ' + str(start))
            st = st + 1
            if st <= 5 :
                continue
            # print ('Start GNSS: ' + str(spress) + ' this GNSS: ' + str(apress) + '\n')
            # if int(spress)+90 <= int(apress) :
                # continue
            stop = atime
            ldist = distance.distance(spnt, bpnt).m
            if ldist > 1500 :
                lo = "LOUT"
            else :
                lo = "HOME"
            fltime = datetime.strptime(stop, FMT) - datetime.strptime(start, FMT)
            # print('Glider: ' + str(gid) + ' Date: ' + str(fdate) + ' Flight Time: ' + str(fltime) + '\n')
            flc =  ''.join(islice(str(fltime), 1)) 
            if flc == '-' :
                res = str(fltime).split(",")
                str(res[1]).split()
                ftime = res[1]
            else :
                ftime = fltime
            # print('Glider: ' + str(gid) + ' Date: ' + str(fdate) + ' Flight Time: ' + str(ftime) + ' Landing: ' + str(lo) + ' Start Alt: ' + str(int(M2F*int(spress))) + ' ft MSL')
            print('Glider: ' + str(gtype) + ' Date: ' + str(fdate) + ' Flight Time: ' + str(ftime) + ' Landing: ' + str(lo) + ' Start Alt: ' + str(int(M2F*int(spress))) + ' ft MSL')
            print('Start Time: ' + str(start) + ' Stop Time: ' + str(stop) + ' Max Altitude: ' + str(int(M2F*int(mpress))) + '[' + str(int(maxaglalt)) + '] ft MSL')
            i = 0
            while (i < len(seng)) : 
                s = seng[i]
                if (i+1 == len(seng)) :
                    ss = atime
                    hgain = int(mslalt) - int(msleng[i])
                else :
                    ss = seng[i+1]
                    hgain = int(msleng[i+1]) - int(msleng[i])
                rntime = datetime.strptime(ss, FMT) - datetime.strptime(s, FMT)
                runtime = round(rntime.total_seconds())/60
                # if int(rpm) > 0 :
                    # sensor = "RPM"
                # else :
                    # sensor = "ENL"
                if (ma.floor(int(runtime)) > 0) :
                    print(gtype + "'s " + sensor + " monitor reports Engine Run " + str(int(runtime)) + " minutes, starts at T=" + str(int(s)) + " and: " + str(int(msleng[i])) + " msl [" + str(int(agleng[i])) + " agl]; Height gain/loss is: " + str(int(hgain)))
                i += 2
                # print("Engine Run ends at T=" + str(ss) + " altitudes are: " + str(int(mslrpm[i])) + " msl  / " + str(int(aglrpm[i])) + " agl")
            if (len(senl) > 0) :
                print(gtype + " Motor noise registered by ENL sensor at t=" + str(senl) + " and " + str(aglenl) + "AGL")
            if (len(smop) > 0) :
                print(gtype + " Motor noise registered by MOP sensor at t=" + str(smop) + " and " + str(aglmop) + "AGL")
            i = 0
            print('\n\n')
            fnout.write(str(fdate) + ',' + str(igcfn) + ',' + str(gtype) + ',' + str(ftime) + 
            ',' + str(start) + ',' + str(stop) + "\n")
            # print('FLight ended, resetting variables for next flight\n')
            HAT = onc = 0
            shat = []
            mhat = []
            that = []
            ahat = []
            atime = alat = alon = aN = aW = aF = apress = agnss = btime = 0
            spd = start = st = spress = bcnt = stop = hpress = mpress = sginit = 0
            eng = rpm = enl = mop = rpmval = enlval = mopval = engflg = enlflg = mopflg = 0
            seng = []
            msleng = []
            agleng = []
            senl = []
            smop = []
            maxaglalt = aglalt = 0
  except Exception as e:
      print(e)
      print('Exception occurred, go to next file')
      # traceback.print_exc()
      return
  # print('Start: ' + str(start) + ' Stop: ' + str(stop))
  if ((stop == 0) and (start != 0)) :
      print('End) of Trace, No stop time found, print anyway')
      stop = atime
      if onc == 2 :
          etime = atime
          ltime = datetime.strptime(atime, FMT) - datetime.strptime(etime, FMT)
          lsec = ltime.total_seconds()
          # print('Glider was below HD for: ' + str(lsec)+ ' seconds, Lowest Alt: ' + str(xpress) + ' at: ' + str(xtime))
          shat.append(lsec)
          mhat.append(xpress)
      # print('Start: ' + str(start) + ' Stop: ' + str(stop))
      fltime = datetime.strptime(stop, FMT) - datetime.strptime(start, FMT)
      ldist = distance.distance(spnt, bpnt).m
      if ldist > 1500 :
          lo = "LOUT"
      else :
          lo = "HOME"
      # print('Glider: ' + str(gid) + ' Date: ' + str(fdate) + ' Flight Time: ' + str(fltime) + '\n')
      flc =  ''.join(islice(str(fltime), 1)) 
      if flc == '-' :
          res = str(fltime).split(",")
          str(res[1]).split()
          ftime = res[1]
      else :
          ftime = fltime
      print('Glider: ' + str(gid) + ' Date: ' + str(fdate) + ' Flight Time: ' + str(ftime) + ' Landing: ' + str(lo) + ' Start Alt: ' + str(int(M2F*int(spress))) + ' ft MSL')
      print('Start Time: ' + str(start) + ' Stop Time: ' + str(stop) + ' Max Altitude: ' + str(int(M2F*int(mpress))) + ' [' + str(int(maxaglalt)) + '] ft MSL')
      i = 0
      while (i < len(seng)) : 
          s = seng[i]
          if (i+1 == len(seng)) :
              ss = atime
              hgain = int(mslalt) - int(msleng[i])
          else :
              ss = seng[i+1]
              hgain = int(msleng[i+1]) - int(msleng[i])
          rntime = datetime.strptime(ss, FMT) - datetime.strptime(s, FMT)
          runtime = round(rntime.total_seconds())/60
          # if int(eng) > 0 :
              # sensor = "RPM"
          # else :
              # sensor = "ENL"
          if (ma.floor(int(runtime)) > 0) :
              print(sensor + " monitor reports Engine Run " + str(int(runtime)) + " minutes, starts at T=" + str(int(s)) + " and: " + str(int(msleng[i])) + " msl [" + str(int(agleng[i])) + " agl]; Height gain/loss is: " + str(int(hgain)))
          i += 2
          # print("Engine Run ends at T=" + str(ss) + " altitudes are: " + str(int(mslrpm[i])) + " msl  / " + str(int(aglrpm[i])) + " agl")
      i = 0
      if (len(senl) > 0) :
          print(gtype + " Motor noise registered by ENL sensor at t=" + str(senl) + " and " + str(aglenl) + "AGL")
      if (len(smop) > 0) :
          print(gtype + " Motor noise registered by MOP sensor at t=" + str(smop) + " and " + str(aglmop) + "AGL")
      # print('\n\n')
      fnout.write(str(fdate) + ',' + str(igcfn) + ',' + str(gtype) + ',' + str(ftime) + 
      ',' + str(start) + ',' + str(stop) + "\n")
if (len(sys.argv) == 1) :
    print("Usage: flt-presalt.py [dir]/ \n")
    sys.exit(0)
fnout = open("Flt-times." + str(os.getpid()) + ".csv", "w")
fnout.write('Date (MM/DD/YYYY),File,Gtype,Flight Time,Start Time,End Time,Landing\n') 

# read in the crop data and get info about it
print("Adding DEM heights for each lat/long")
demdata = rio.open('conus.tif')
dband1 = demdata.read(1)

for f in sys.argv :
  cmd = "ls " + str(f) + "*.IGC"
  # print(cmd)
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



