import matplotlib.pyplot as plt
import mne
import numpy as np
from scipy.signal import welch
from scipy.io import savemat
def loaddata(filename):
    """
    get raw gdf data from file called 'filename'
    """
    with open(filename, 'rb') as handle:
        data = mne.io.read_raw_gdf(filename, stim_channel="auto", verbose='ERROR', 
                                   exclude=(["EOG-left", "EOG-central", "EOG-right"]),preload=True)
    return data

trails = []
labels = []
tmin,tmax = 0,4
num_sbj = 9
for i in range(1,num_sbj+1):
    filename = f"D:\Study\Dissertation\Datasets\BCICIV_2a_gdf\A0{i}T.gdf"
    raw_gdf = loaddata(filename)
    # raw_gdf.filter(7, 47., fir_design='firwin')
    eventtime,etype = mne.events_from_annotations(raw_gdf)
    data = raw_gdf.get_data() # data size (22,672528)
    fs = int(raw_gdf.info['sfreq']) # 250
    trail = []
    label = []
    # event_id = [etype['769'],etype['770'],etype['771'],etype['772']] # get event id
    event_id = [etype['769'],etype['770']]
    # subject 4 is different, its id is {'1023': 1, '1072': 2, '32766': 3, '768': 4, '769': 5, '770': 6, '771': 7, '772': 8}
    # 769,770,771,772 correspond to 5,6,7,8
    # others are 7,8,9,10
    for j in range(len(eventtime)):
        if eventtime[j][2] in event_id:
            trail.append(data[:,eventtime[j][0]+fs*tmin:eventtime[j][0]+fs*tmax])
            label.append(eventtime[j][2]-event_id[0])
    trails.append(trail)
    labels.append(label)
trails = np.array(trails)
labels = np.array(labels)
channelname = np.array(raw_gdf.ch_names)
sub_d = trails[0]
print(sub_d)
sub_d = sub_d.reshape(288,22,500)
print(sub_d)
# d_s1 = trails[0].reshape()
d_s1 = trails[0][0]
for i in range(1,trails[0].shape[0]):
    d_s1=np.concatenate((d_s1, trails[0][i]), axis=1)
plt.plot(d_s1[0,0:1000])
plt.show()
# print(d_s1.shape)
# print(trails[0][1])
# print(d_s1[:,1000:2000])
# savemat('D:\Study\Dissertation\Datasets\matdata.mat', {'data':d_s1})
# np.savez('D:\Study\Dissertation\Datasets\EData2type1000.npz', data = trails, labels = labels, channels = channelname)
print(trails.shape,labels.shape)

def getPSD(trails,fs,fmin,fmax):
    PSDset = []
    for i in trails:
        psdset = []
        for j in i:
            f_P, Pxx = welch(j,
                        fs=fs, 
                        window='hann', 
                        noverlap=None,
                        detrend='constant',
                        scaling='spectrum',
                        axis=- 1, 
                        average='mean')
            psdset.append(Pxx[fmin:fmax])
        PSDset.append(psdset)
    return PSDset,f_P[fmin:fmax]

PSD_all = []
# trails size (9,288,22,750)
# for i in trails:
#     # i size (288,22,750)
#     psd,fp = getPSD(i,fs,fmin=5,fmax=45)
#     PSD_all.append(psd)
# PSD_all = np.array(PSD_all)
# f_P = np.array(fp)
# np.savez('D:\Study\Dissertation\Datasets\PSDdata.npz', data = PSD_all, f = f_P)