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
#dirz = glob.glob('trial-easyhardeasy*')
dirz = glob.glob('trial-fixedsize*')
dirz2 = glob.glob('trial-ref*-HARD-*')

dirz = dirz + dirz2
dirz.sort()
NBPLOTS = len(dirz)
SS = np.ceil(np.sqrt(NBPLOTS))
linez=[]

plt.figure(1,  figsize=(3, 3), dpi=100, facecolor='w', edgecolor='k')

nplot = 1
thards= []
teasys=[]
colorz=['b', 'r', 'g', 'm', 'c', 'orange']
labelz=['10 neurons', '100 neurons', '27 neurons', '50 neurons', 'Variable Size']
for (num, droot) in enumerate(dirz):
    t = []
    for v in range(20):
        dfull = droot + "/v" + str(v)
        t.append(np.loadtxt(dfull+"/test.txt")[:200, :])
    t = np.dstack(t)
    tmean = np.mean(t, axis=2)
    tstd = np.std(t, axis=2)
    tmedian = np.median(t, axis=2)
    tq25 = np.percentile(t, 25, axis=2)
    tq75 = np.percentile(t, 75, axis=2)
    
    for vari in [2]:  # range(2, tmean.shape[1]):
        #plt.fill_between(range(tmean.shape[0]), tq25[:, vari], tq75[:, vari], linewidth=0.0, alpha=0.3, facecolor=colorz[vari])
        if num == len(dirz)-1: # The last curve is that of the variable-size runs
            linez.append(plt.plot(tmedian[:, vari], color='k', linewidth=2, label=labelz[num]))
        else:
            linez.append(plt.plot(tmedian[:, vari], color=colorz[num], label=labelz[num]))
    plt.axis([0, tmean.shape[0], 0, 50])

    print num, tmean[90, :], tmean[190, :], tmean[-1, :], droot
    thards.append(tmean[90,:])
    teasys.append(tmean[-1,:])

    nplot += 1

plt.xlabel('Iterations (x1000)')
plt.ylabel('Loss')
plt.legend(fontsize=9)
plt.tight_layout()

print "Data read."

plt.show()

plt.savefig('figFS.png', bbox_inches='tight')
