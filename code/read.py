import os

from obspy import read, read_inventory
from obspy.core.utcdatetime import UTCDateTime
from obspy.clients.fdsn import Client

def load_data(loc, channels, starttime, endtime, data_directory):

    mseed_file = '{0:s}/data/{1:s}_{2:s}_{3:s}_{4:s}_{5:s}.mseed'.format(data_directory,
                                                                         loc['network'],
                                                                         loc['station'],
                                                                         loc['location'],
                                                                         starttime,
                                                                         endtime)
    xml_file = '{0:s}/inventories/{1:s}_{2:s}_{3:s}_{4:s}_{5:s}.xml'.format(data_directory,
                                                                            loc['network'],
                                                                            loc['station'],
                                                                            loc['location'],
                                                                            starttime,
                                                                            endtime)

    if not os.path.exists(data_directory):
        os.makedirs(data_directory)
        os.makedirs(data_directory+'/data')
        os.makedirs(data_directory+'/inventories')
        
    if os.path.isfile(mseed_file) and os.path.isfile(xml_file):
        print('Reading data from local files')
        st = read(mseed_file)
        inv = read_inventory(xml_file, format='stationxml')
    else:
        print('Collecting data from IRIS')
        # Fetch waveform from IRIS FDSN web service into a ObsPy stream object
        # and automatically attach correct response
        t1 = UTCDateTime(starttime)
        t2 = UTCDateTime(endtime)

        fdsn_client = Client('IRIS')
        st = fdsn_client.get_waveforms(network=loc['network'], station=loc['station'],
                                       location=loc['location'], channel=channels,
                                       starttime=t1, endtime=t2)
        inv = fdsn_client.get_stations(network=loc['network'], station=loc['station'],
                                       location=loc['location'], channel=channels,
                                       starttime=t1, endtime=t2,
                                       level='response')

        # write mseed and inventory files
        st.write(mseed_file, format='MSEED')
        inv.write(xml_file, format='stationxml')

    st.attach_response(inv)
    return (st, inv)
