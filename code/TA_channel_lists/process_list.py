import numpy as np
import matplotlib.pyplot as plt

VHZ_stations = [line.rstrip('\n').split('\t') for line in open('VHZ.txt') if line[0] != '#']
LDO_stations = [line.rstrip('\n').split('\t') for line in open('LDO.txt') if line[0] != '#']
LDV_stations = [line.rstrip('\n').split('\t') for line in open('LDV.txt') if line[0] != '#']

stations = {}

for sta_list in [VHZ_stations, LDO_stations, LDV_stations]:
    for sta in sta_list:
        stations[sta[1]] = {}

for sta_list in [VHZ_stations, LDO_stations, LDV_stations]:
    for sta in sta_list:
        stations[sta[1]][sta[3]] = sta

# NETWORK  STATION   LOCATION	CHANNEL	STARTTIME	       ENDTIME	               LAT	 LON	     ELEVATION	DEPTH
# 'TA',    'I17K',   '00',      'VHZ',  '2017-06-09 00:00:00', '2599-12-31 23:59:59', '63.8864', '-160.695', '105', '2'

LDV_VHZ_stations = {station: {'net': channels['VHZ'][0],
                              'sta': channels['VHZ'][1],
                              'loc': channels['VHZ'][2],
                              'lat': float(channels['VHZ'][6]),
                              'lon': float(channels['VHZ'][7]),
                              'VHZ': {'start': channels['VHZ'][4],
                                      'end': channels['VHZ'][5],
                                      'elevation': float(channels['VHZ'][8]),
                                      'depth': float(channels['VHZ'][9])},
                              'LDV': {'start': channels['LDV'][4],
                                      'end': channels['LDV'][5],
                                      'elevation': float(channels['LDV'][8]),
                                      'depth': float(channels['LDV'][9])}}
                    for (station, channels) in stations.items() if 'LDV' in channels and 'VHZ' in channels}

LDO_VHZ_stations = {station: {'net': channels['VHZ'][0],
                              'sta': channels['VHZ'][1],
                              'loc': channels['VHZ'][2],
                              'lat': float(channels['VHZ'][6]),
                              'lon': float(channels['VHZ'][7]),
                              'VHZ': {'start': channels['VHZ'][4],
                                      'end': channels['VHZ'][5],
                                      'elevation': float(channels['VHZ'][8]),
                                      'depth': float(channels['VHZ'][9])},
                              'LDO': {'start': channels['LDO'][4],
                                      'end': channels['LDO'][5],
                                      'elevation': float(channels['LDO'][8]),
                                      'depth': float(channels['LDO'][9])}}
                    for (station, channels) in stations.items() if 'LDO' in channels and 'VHZ' in channels}

LDO_locs = np.array([[sta['lat'], sta['lon']] for (name, sta) in LDO_VHZ_stations.items()]).T
LDV_locs = np.array([[sta['lat'], sta['lon']] for (name, sta) in LDV_VHZ_stations.items()]).T

plt.scatter(LDO_locs[1], LDO_locs[0])
plt.scatter(LDV_locs[1], LDV_locs[0])
plt.show()


#print ([LDO_VHZ_stations[name] for (name, sta) in LDV_VHZ_stations.items() if sta['lat'] < 50.])
print ([name for (name, sta) in LDO_VHZ_stations.items() if sta['lat'] < 50.])
