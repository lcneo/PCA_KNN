import scineo as sn
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
def show_max(mat):
    for i in mat.columns:
        for j in mat.index:
            if mat[i][j] == mat.max().max():
                print("n = %s\tk = %s\t有最大识别率:%.2f%%"%(j,i,mat.max().max()*100))

#读取文件
train,train_y,test,test_y = np.load("DataSet/No4.npy")
#将图片拉升成一行
train_x,test_x = train.reshape(-1,128*128),test.reshape(-1,128*128)
#将样本合并成一个矩阵进行pca
X = np.concatenate([train_x,test_x])
start = time.time()
l_1 = []
pca_number = 1892
#i为pca提取后的维度数
for i in range(3,100):
    mat = sn.pca(X,n_components=i)
    train_x,test_x = mat[:len(train_y)],mat[len(train_y):]
    detail = mat.shape[1]
    l_2 = []
    #j为knn中k的值
    print("")
    print("pca = %f\tdetail = %d"%(i/pca_number,i))
    ac = sn.predict(train_x,train_y,test_x,test_y,model = sn.model_svm,show=True)
    l_1.append(ac)

print("\ntiem = %.2fS"%(time.time() - start))
pk = pd.DataFrame(l_1)
#pk.columns = list(range(1,31))
pk.index = list(range(3,100))
pk.to_csv("DataSet/svm.csv")
print("训练样本数:%d\t测试样本数:%d"%(len(train_x),len(test_y)))
show_max(pk)
print(pk)

