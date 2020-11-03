import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure(figsize=(6, 5))
ax = fig.add_subplot(111)

yhat = np.linspace(0,1,100)
cost0 = -np.log2(1-yhat)
cost1 = -np.log2(yhat)

ax.plot(yhat, cost0, yhat, cost1)
ax.legend(['True class y=0', 'True class y=1'])
ax.set_xlabel(r'$\hat{y}$')
ax.set_ylabel('Cross-entropy loss')
fig.savefig('graph.png') 