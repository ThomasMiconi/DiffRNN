import glob
import numpy as np
import matplotlib.pyplot as plt

font = {#'family' : 'normal',
#                'weight' : 'bold',
                        'size'   : 10}
plt.rc('font', **font)

plt.ion()
np.set_printoptions(precision=3, suppress=True)

#dirz = glob.glob('trial-max10*')
#dirz = glob.glob('trial-new-hardeasy*')
dirz = glob.glob('trial-easyhardeasy*')
#dirz = glob.glob('trial-fixedsize*')
#dirz = glob.glob('trial-ref*EASYHARDEASY*')
dirz.sort()
NBPLOTS = len(dirz)
SS = np.ceil(np.sqrt(NBPLOTS))

plt.figure(1,  figsize=(4, 3), dpi=100, facecolor='w', edgecolor='k')

nplot = 1
thards= []
teasys=[]
colorz=['b', 'b', 'b', 'r', 'g']
for (num, droot) in enumerate(dirz):
    t = []
    for v in range(10):
        dfull = droot + "/v" + str(v)
        t.append(np.loadtxt(dfull+"/test.txt"))
    t = np.dstack(t)
    tmean = np.mean(t, axis=2)
    tstd = np.std(t, axis=2)
    tmedian = np.median(t, axis=2)
    tq25 = np.percentile(t, 25, axis=2)
    tq75 = np.percentile(t, 75, axis=2)
    
    ax = plt.subplot(SS, SS, nplot)
    ax.set_title(num)
    for vari in [3, 2]:  # range(2, tmean.shape[1]):
        plt.fill_between(range(tmean.shape[0]), tq25[:, vari], tq75[:, vari], linewidth=0.0, alpha=0.3, facecolor=colorz[vari])
        plt.plot(tmedian[:, vari], color=colorz[vari])
    plt.axis([0, tmean.shape[0], 0, 50])

    print num, tmean[90, :], tmean[190, :], tmean[-1, :], droot
    thards.append(tmean[90,:])
    teasys.append(tmean[-1,:])

    nplot += 1

print "Data read."

plt.show()

#plt.savefig('fig1.png', bbox_inches='tight')
