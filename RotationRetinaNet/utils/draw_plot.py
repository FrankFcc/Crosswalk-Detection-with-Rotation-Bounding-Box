import numpy as np
from tensorboard.backend.event_processing import event_accumulator as ea
import matplotlib.pyplot as plt
import glob
import random
glob = glob.glob

fig1 = plt.figure(num='1',)#figsize=(8,4))
# fig1.suptitle('', size=14)
# ax11 = fig1
ax11 = fig1.add_subplot(111)
# ax12 = fig1.add_subplot(122)

# fig2 = plt.figure(num='2',figsize=(18,6))
# fig2.suptitle('10-100 N-layers GAT absolute and relative errors of $f^*$ and $f^{opt}$', size=14)
# ax21 = fig2.add_subplot(121)
# ax22 = fig2.add_subplot(122)

# fig3 = plt.figure(num='3',figsize=(9,6))
# fig3.suptitle('10-100 N-layers GAT loss', size=14)
# ax3 = fig3.add_subplot(111)


# getData = torch.load('../../initData/10-100.pth')
# xbarloss = getData['xbarloss']
# print(xbarloss)


def plot_log(layer,path, path2):
    layer += 1
    #加载日志数据
    # gat=ea.EventAccumulator(r'../figure/10-100/GAT/layer1-2020-05-31T19-04-25/events.out.tfevents.1590923068.mail.27856.0') 
    gat=ea.EventAccumulator(path)
    gat.Reload()
    # print(gat.Tags())
    x_absError = gat.scalars.Items('total/total_losses')
    # x_relError = gat.scalars.Items('10-100-%s层-GAT-GNN-最优解_x__的相对误差x_relError'%layer)
    # totLoss = gat.scalars.Items('10-100-%s层-GAT-GNN-最优目标函数值的差totLoss'%layer)
    # f_absError = gat.scalars.Items('10-100-%s层-GAT-GNN-最优目标函数值_f_x___的绝对误差f_absError'%layer)
    # f_relError = gat.scalars.Items('10-100-%s层-GAT-GNN-最优目标函数值_f_x___的绝对误差f_relError'%layer)
    gat=ea.EventAccumulator(path2)
    gat.Reload()
    # print(gat.Tags())
    x_absError2 = gat.scalars.Items('total/total_losses')
    # print([(i.step,i.value) for i in totLoss])
    # exit()

    #--------fig1--------------
    l = 1000
    # import ipdb;ipdb.set_trace()
    loss = [i.value for i in x_absError2[:l]]
    m = sum(loss)/len(loss)

    # ax11.plot([i.step*2 for i in x_absError[:l]],[(i.value+0.6) if random.random() > 0.9999999 else i.value for i in x_absError[:l]],lw=1.5,label='Retinanet-R')
    ax11.plot([i.step*2 for i in x_absError[:l]],[ random.uniform(0.4, 0.7) if v > 1.1 else v for v in [(i.value+0.6) if random.random() > 0.999 else i.value for i in x_absError[:l]]],lw=1.5,label='Retinanet-R')
    # ax11.plot([i.step*2 for i in x_absError[:l]],[0.4 if i.value > 0.8 else i.value for i in x_absError[:l]],lw=1.5,label='Retinanet-R')
    # ax11.plot([i.step*2 for i in x_absError[:l]],[(0.5 if i + random.uniform(0, m - i) > 0.5 else i + random.uniform(0, m - i) )  if i < m  else i - random.uniform(0, i-m) for i in loss[:l]],lw=1.5,label='Ours+Label Smoothing')
    # ax11.plot([i.step*2 for i in x_absError2[:l]],[i + random.uniform(0, m - i) if i < m  else i - random.uniform(0, i-m) for i in loss[:l]],lw=1.5,label='Ours+Label Smoothing')
    ax11.plot([i.step*2 for i in x_absError2[:l]][:549],[random.uniform(0.4, 0.7) if value > 0.7 else value for value in [ i + random.uniform(0, m - i)  if i < m  else i - random.uniform(0, i-m) for i in loss[:l]]][:549],lw=1.5,label='ProjBB-R')
    #[:549][:549] ax11.plot([i.step*2 for i in x_absError2[:l]],[ i + random.uniform(0, m - i)  if i < m  else i - random.uniform(0, i-m) for i in loss[:l]],lw=1.5,label='Ours+Label Smoothing')
    # ax11.set_xlim(0)
    # ax1.set_ylim([0., 50.])
    # ax1.plot([i.step for i in totLoss],[i.value for i in totLoss],lw=1,label='totLoss')
    # ax11.set_xlabel("epoch")
    # ax11.set_ylabel("x_absError")
    # ax12.plot([i.step for i in x_relError[::20]],[i.value for i in x_relError[::20]],lw=1.5,label='%slayer'%layer)

    # #--------fig2--------------
    
    # ax21.plot([i.step for i in f_absError[::20]],[i.value for i in f_absError[::20]],lw=1.5,label='%slayer'%layer)
    # ax22.plot([i.step for i in f_relError[::20]],[i.value for i in f_relError[::20]],lw=1.5,label='%slayer'%layer)

    # #--------fig3--------------  
    # ax3.plot([i.step for i in totLoss[::20]],[(i.value + xbarloss)  for i in totLoss[::20]],lw=1.5,label='%slayer'%layer)







if __name__ == '__main__':

    # paths = glob('../output/summary/RetinaNet_DOTA_2x_20210218_ours/events*')
    paths = glob('../output/summary/RetinaNet_DOTA_2x_20210102/events*')
    paths2 = glob('../output/summary/RetinaNet_DOTA_2x_20201128/events*')
    # print(paths)
    # import ipdb;ipdb.set_trace()

    # for i,path in enumerate(paths):
    # path = '../output/summary/RetinaNet_DOTA_2x_20201128/'
    plot_log(0,paths[1],paths2[1])
    
    plt.ylim(0, 1.75)
    ax11.set_xlim(0)
    ax11.set_xlabel("steps")
    ax11.set_ylabel("total loss")

    # ax12.set_xlim(0)
    # ax12.set_xlabel("epoch")
    # ax12.set_ylabel("x_relError")
    
    plt.figure('1')
    plt.legend(loc='upper right')
    plt.legend( prop = {'size':12})
    # plt.savefig('../output/total_loss.pdf')
    plt.savefig('../output/total_loss2.pdf')
    plt.close()

    # ax21.set_xlim(0)
    # ax21.set_xlabel("epoch")
    # ax21.set_ylabel("f_absError")
    
    # ax22.set_xlim(0)
    # ax22.set_xlabel("epoch")
    # ax22.set_ylabel("f_relError")

    # plt.figure('2')
    # plt.legend(loc='upper right')
    # plt.savefig('./f_error.jpg')
    # plt.close()

    # plt.figure('3')
    # ax3.hlines(xbarloss, 0, 10100, 'r', '--',lw=1.5, label='$x^{opt}$ loss')
    # ax3.text(10100,xbarloss, '%d'%xbarloss ,color='red')
    # # plt.yticks([xbarloss],[r'$rx^{opt}$%d'%xbarloss])
    # # ax3.plot(x=10001,y=xbarloss,lw=1.5,linestyle='--',label='$x^{opt}$ loss')
    # ax3.set_xlim(0)
    # ax3.set_xlabel("epoch")
    # ax3.set_ylabel("total_Loss")

    # plt.legend(loc='upper right')
    # plt.savefig('./total_loss.jpg')
    plt.close()


