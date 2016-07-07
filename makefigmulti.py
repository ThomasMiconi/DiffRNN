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
#dirz = glob.glob('trial-EHE*MULTIPGRAD-1*')
dirz = glob.glob('trial-EHE*900000*')
#dirz = glob.glob('trial-fixedsize*')
#dirz = glob.glob('trial-ref*EASYHARDEASY*')
dirz.sort()
NBPLOTS = len(dirz)
SS = np.ceil(np.sqrt(NBPLOTS))

plt.figure(1,  figsize=(4, 3), dpi=100, facecolor='w', edgecolor='k')

nplot = 1
perfs = []
nbneurs = []
dirs = []
colorz=['b', 'b', 'b', 'r', 'g']
for (num, droot) in enumerate(dirz):
    t = []
    for v in range(10):
        dfull = droot + "/v" + str(v)
        t.append(np.loadtxt(dfull+"/output.txt"))
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

    p1 = int(tmean.shape[0] / 3)
    p2 = 2*int(tmean.shape[0] / 3)
    p3 = -1

    print num, tmean[p1, :], tmean[p2, :], tmean[p3, :], droot
    perfs.append([tmean[p1,2], tmean[p2, 2], tmean[p3, 2]])
    nbneurs.append([tmean[p1,3], tmean[p2, 3], tmean[p3, 3]])
    dirs.append(droot)

    nplot += 1

print "Data read."

perfs = np.array(perfs)
p = perfs[:,1]
nbneurs = np.array(nbneurs)
dneur = nbneurs[:, 1] - nbneurs[:,2]
ord = np.argsort(p)
data = np.vstack((ord, dneur[ord], p[ord])).T


plt.show()

#plt.savefig('fig1.png', bbox_inches='tight')
