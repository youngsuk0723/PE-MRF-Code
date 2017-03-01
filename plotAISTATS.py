

import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as pl
import numpy as np
mpl.rcParams['ps.useafm'] = True
mpl.rcParams['pdf.use14corefonts'] = True
mpl.rcParams['text.usetex'] = True


FPR_H_W = [0 , 0.01, 0.1, 0.41, 0.58, 0.66, 0.71, 0.8, 0.95, 1]
TPR_H_W = [0, 0.21, 0.42, 0.7, 0.81, 0.86, 0.89, 0.93, 0.98, 1]

FPR_L_W = [0, 0.07, 0.41, 0.58, 0.68 ,0.73 , 0.8 , 0.9, 1]
TPR_L_W= [0, 0.1,  0.46, 0.63, 0.71, 0.76, 0.82, 0.9, 1]


#LowSparse
TPR_L = [1,9.070636981516848607e-01, 8.946462300259729883e-01, 8.663806025026936641e-01, 7.797747002937570615e-01, 6.139004769951745821e-01, 3.534513844963888540e-01, 1.787779975120244724e-01,0]
FPR_L = [1,8.754667738911369179e-01, 8.614909840618097947e-01, 8.241403904403589165e-01, 7.267765401050229057e-01, 5.170980804155452804e-01, 2.400105818474775887e-01, 5.934007664026325762e-02,0]

#High Sparse                                                                              
TPR_H = [1,9.538227452696735886e-01,9.513985028454313708e-01,9.409195836916376399e-01,9.143659708837504896e-01,8.577147718331913362e-01,7.407982238398482799e-01,5.442009396911233754e-01,0]

FPR_H = [1,8.816720219709964779e-01,8.658402016719904948e-01,8.239803149771619539e-01,7.225428437861363129e-01,5.157032950055980836e-01,2.377254039162460530e-01,6.530052048408768739e-02,0]
                                       

pl.figure(figsize=(4,6))
pl.subplot(2, 1, 1)
pl.plot(FPR_H,TPR_H,'ro-', label = 'PE-MRF')
pl.plot(FPR_H_W,TPR_H_W,'gs-', label = 'VS-MRF')
pl.plot([0,1],[0,1],'--')
# pl.xlabel('FPR')
pl.ylabel('TPR')
pl.title('High Sparsity Structure')
pl.legend(loc = 'lower right')
# pl.show()
# pl.rcParams.update({'font.size': 12})
# pl.savefig('highSparse.png', bbox_inches = 'tight', dpi = 1000)


pl.subplot(2, 1, 2)
pl.plot(FPR_L,TPR_L,'ro-', label = 'PE-MRF')
pl.plot(FPR_L_W,TPR_L_W,'gs-', label = 'VS-MRF')
pl.plot([0,1],[0,1],'--')
pl.xlabel('FPR')
pl.ylabel('TPR')
pl.title('Low Sparsity Structure')


#pl.plot(FPR_H,TPR_H,'bo-')
#pl.plot(FPR_H1,TPR_H1,'ro-')
#pl.plot(FPR_H_W,TPR_H_W,'ws-')
pl.legend(loc = 'lower right')
# pl.savefig('lowSparse.png',  bbox_inches = 'tight', dpi = 1000)
    
pl.savefig('AISTATS_plots.eps', format = 'eps', bbox_inches = 'tight', dpi = 1000)
# pl.show()