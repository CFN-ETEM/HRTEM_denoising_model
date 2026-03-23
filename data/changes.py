import numpy as np

smmat=[3,4,5]
gmat=[2,5,8]

for i1 in range(3):
    for i2 in range(3):
        snow=smmat[i1]
        gnow=gmat[i2]
        datanow=np.load(f'simulated_edose_1e{snow}_sigma_0p{gnow}.npy')
        data1=datanow[:10,:,:]
        data2=datanow[10:,:,:]
        np.save(f'simulated_edose_1e{snow}_sigma_0p{gnow}_part0.npy',data1)
        np.save(f'simulated_edose_1e{snow}_sigma_0p{gnow}_part1.npy',data2)
        print(data1.shape,data2.shape)