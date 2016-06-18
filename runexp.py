# Submit jobs to the cluster. 

# /opt/python-2.7.10/bin/python


import sys
import os
import shutil

"""
g = {
'COEFFMULTIPNORM' : 3e-5,
'DELETIONTHRESHOLD': .01,
'MINMULTIP': .01*.25,  # Must be lower than DELETIONTHRESHOLD !
'NBMARGIN' : 1,
'PROBADEL': .003,
'PROBAADD': .1,
'RNGSEED' : 0
}
"""
# min-char-rnn, 250K, hidden_size = 30: loss 12.86
allopts = [

        #"EXPTYPE EASY COEFFMULTIPNORM 3e-5 NBMARGIN 1 DELETIONTHRESHOLD .05 MINMULTIP .025 PROBADEL .5 PROBAADD .05 NBSTEPS 300000",  
        #"EXPTYPE EASY COEFFMULTIPNORM 3e-5 NBMARGIN 1 DELETIONTHRESHOLD .05 MINMULTIP .025 PROBADEL .25 PROBAADD .05 NBSTEPS 300000",  # Ref
        #"EXPTYPE EASY COEFFMULTIPNORM 3e-5 NBMARGIN 1 DELETIONTHRESHOLD .05 MINMULTIP .025 PROBADEL .25 PROBAADD .025 NBSTEPS 300000",  
        #"EXPTYPE EASY COEFFMULTIPNORM 5e-5 NBMARGIN 1 DELETIONTHRESHOLD .05 MINMULTIP .025 PROBADEL .5 PROBAADD .05 NBSTEPS 300000",  
        #"EXPTYPE EASY COEFFMULTIPNORM 3e-5 NBMARGIN 2 DELETIONTHRESHOLD .05 MINMULTIP .025 PROBADEL .5 PROBAADD .05 NBSTEPS 300000",  
        #"EXPTYPE EASY COEFFMULTIPNORM 3e-5 NBMARGIN 1 DELETIONTHRESHOLD .05 MINMULTIP .025 PROBADEL 1.0 PROBAADD .05 NBSTEPS 300000",  
        #"EXPTYPE EASY COEFFMULTIPNORM 3e-5 NBMARGIN 2 DELETIONTHRESHOLD .05 MINMULTIP .025 PROBADEL 1.0 PROBAADD .05 NBSTEPS 300000",  

        #"HIDDENSIZE 10 NBSTEPS 300000",  
        #"HIDDENSIZE 30 NBSTEPS 300000",  
        "HIDDENSIZE 27 NBSTEPS 300000",  
        #"HIDDENSIZE 50 NBSTEPS 300000",  
        #"HIDDENSIZE 70 NBSTEPS 300000",  
        #"HIDDENSIZE 100 NBSTEPS 300000",  

        #"EXPTYPE EASY COEFFMULTIPNORM 3e-5 NBMARGIN 1 DELETIONTHRESHOLD .05 MINMULTIP .025 PROBADEL .25 PROBAADD .05 NBSTEPS 300000",
        #"EXPTYPE HARD COEFFMULTIPNORM 3e-5 NBMARGIN 1 DELETIONTHRESHOLD .05 MINMULTIP .025 PROBADEL .25 PROBAADD .05 NBSTEPS 300000",
        #"EXPTYPE EASYHARDEASY COEFFMULTIPNORM 3e-5 NBMARGIN 1 DELETIONTHRESHOLD .05 MINMULTIP .025 PROBADEL .25 PROBAADD .05 NBSTEPS 300000"
        ]


for optionz in allopts:

    #dirname = "trial-easyonly-CMN-" + optionz.replace(' ', '-')
    dirname = "trial-fixedsize-CMN-" + optionz.replace(' ', '-')
    #dirname = "trial-ref-CMN-" + optionz.replace(' ', '-')

    if os.path.exists(dirname):
        shutil.rmtree(dirname)
    os.mkdir(dirname)
    os.chdir(dirname)
    print os.getcwd()

    for v in range(20):
        os.mkdir("v"+str(v))
        os.chdir("v"+str(v))
        #CMD = "bsub -q short -W 3:00 -g /rnn /opt/python-2.7.10/bin/python ../../rnn.py " + optionz + " RNGSEED " + str(v)
        CMD = "bsub -q short -W 6:00 -g /rnn /opt/python-2.7.10/bin/python ../../min-char-rnn-param.py " + optionz + " RNGSEED " + str(v)
        #print CMD
        retval = os.system(CMD)
        print retval
        os.chdir('..') 
    
    os.chdir('..') 


    #print dirname
    #for RNGSEED in range(2):
    #st = "python rnn.py COEFFMULTIPNORM " + str(CMN) + " DELETIONTHRESHOLD " + str(DT) + " MINMULTIP " \
    #+ str(MMmultiplierofDT*DT) + " PROBADEL " + str(PD) + " PROBAADD " + str(PAmultiplierofPD * PD) \
    #+ " RNGSEED " + str(RNGSEED) + " NUMBERMARGIN " + str(NM)




