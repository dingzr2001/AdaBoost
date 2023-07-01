import numpy as np
from main import Adaboost
test1 = Adaboost(base = 1)      # 0为对率回归，1为决策树桩
test1.fit('data.csv','targets.csv')
predic = test1.predict('data.csv')
np.savetxt("test.csv",test1.predict('data.csv'),delimiter=',')
Y = np.genfromtxt('targets.csv',float,delimiter=',')
Y = Y.reshape(-1,1)
error = np.zeros((predic.shape[0],1))
error[predic==Y] = 1
print(error.sum()/predic.shape[0])

target = np.genfromtxt('targets.csv')
base_list = [1, 5, 10, 100]

for base_num in base_list:
    acc = []
    for i in range(1, 11):
        fold = np.genfromtxt('experiments/base%d_fold%d.csv' % (base_num, i), delimiter=',', dtype=np.int)
        accuracy = sum(target[fold[:, 0] - 1] == fold[:, 1]) / fold.shape[0]
        acc.append(accuracy)
    
    print(np.array(acc).mean())
    