from numpy import *
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.font_manager import *
import csv
import time
def loadDataSet():                # 加载测试数据
         # 数据集返回的是矩阵类型，标签返回的是列表类型\
    #-----------------------文件IO-------------------------
    dataList = loadtxt('./data.csv',float,delimiter=',')
    labelList = loadtxt('./targets.csv',float,delimiter=',')
    X = dataList
    return X,labelList

class Adaboost:
    classArr = None
    def __init__(self,base):
        self.base = base

    def kFoldSplit(self,X, labelList):
        wholeIndexList = array([i for i in range(X.shape[0])])
        selectedIndexList = random.choice(wholeIndexList, size = int(floor(X.shape[0]/10)), replace = False)
        #print(selectedIndexList)
        leftIndexList = delete(wholeIndexList, selectedIndexList)
        testMat = X[selectedIndexList]
        testLabelList = labelList[selectedIndexList]
        trainMat = X[leftIndexList]
        trainLabelList = labelList[leftIndexList]
        return trainMat, trainLabelList, selectedIndexList, testMat, testLabelList

    def stumpClassify(self,X,dimen,threshVal,classify_method): # dimen:第dimen列，也就是第几个特征, threshVal:是阈值  threshIneq：标志
        m = shape(X)[0]
        retArray = ones((m,1))   # 创造一个 样本数×1 维的array数组
        if classify_method == 'lt':        # lt表示less than，表示分类方式，对于小于等于阈值的样本点赋值为0
            retArray[X[:,dimen] <= threshVal] = 0.0
        else:  # 我们确定一个阈值后，有两种分法，一种是小于这个阈值的是正类，大于这个值的是负类，
            #第二种分法是小于这个值的是负类，大于这个值的是正类，所以才会有这里的if 和else
            retArray[X[:,dimen] > threshVal] = 0.0
        return  retArray            # 返回的是一个基分类器的分类好的array数组,是一个列向量
    
    def buildStump(self,X,classLabels,D):
        labelMat = classLabels.reshape(-1,1)#转化为列向量
        #print(shape(labelMat))
        m,n = shape(X)
        #numStemp = 10
        bestStump = {}
        bestClassEst = zeros((m,1))
        minError = inf                      # 无穷
        for i in range(n):                  # 遍历特征
            rangeMin = X[:,i].min()    # 检查到该特征的最小值
            rangeMax = X[:,i].max()
            sortedValue = sort(X[:,i])
            left = 0
            right = 299
            #stepSize = (rangeMax - rangeMin)/numStemp  # 寻找阈值的步长是最大减最小除以10,你也可以按自己的意愿设置步长公式
            while right < m:
                #print(right)
                threshVal = (sortedValue[left] + sortedValue[right]) / 2
                for inequal in ['lt', 'gt']:   # 因为确定一个阈值后，可以有两种分类方式     
                    predictedVals = self.stumpClassify(X,i,threshVal,inequal)  # 确定一个阈值后，计算它的分类结果，predictedVals就是基分类器的预测结果，是一个m×1的array数组
                    errArr = ones((m,1))#m维列向量
                    errArr[predictedVals==labelMat] =0   # 预测值与实际值相同，误差置为0
                    #print(type(errArr))
                    #print(shape(D),shape(errArr))
                    weightedError = dot(D.reshape(1,-1), errArr)     # D就是每个样本点的权值，随着迭代，它会变化，这段代码是误差率的公式
                    #print(type(weightedEroor))
                    if weightedError<minError:     # 选出分类误差最小的基分类器
                        minError=weightedError     # 保存分类器的分类误差
                        bestClassEst = predictedVals.copy()   # 保存分类器分类的结果
                        bestStump['dim']=i          # 保存那个分类器的选择的特征
                        bestStump['thresh']=threshVal    # 保存分类器选择的阈值
                        bestStump['ineq']=inequal        # 保存分类器选择的分类方式
                left += 300
                right += 300
        return bestStump,minError,bestClassEst
    
    
    def adaBoostTrainWithStump(self,X, classLabels,epoch=10):   # 迭代10次，直至误差满足要求，或达到40次迭代
        weakClassArr = []   # 保存每个基分类器的信息，存入列表
        m = shape(X)[0]
        D = ones((m,1))/m   #列向量
        aggClassEst = zeros((m,1))
        for i in range(epoch):
            bestStump,error,classEst = self.buildStump(X,classLabels,D)
            alpha = float(0.5 * log((1.0-error)/max(error,1e-16)))   # 对应公式 a = 0.5* (1-e)/e
            bestStump['alpha']=alpha
            weakClassArr.append(bestStump)  # 把每个基分类器存入列表
            expon =  alpha * ((classLabels.reshape(-1,1) != classEst)*2-1).reshape(-1,1)   # 预测结果与实际值相等则减小权重，否则增加
            D = D * exp(expon)       # 根据公式 w^m+1 = w^m (e^-a*y^i*G)/Z^m
            D = D/D.sum()         # 归一化
            aggClassEst += alpha * classEst   # 分类函数 f(x) = a1 * G1
        return weakClassArr,aggClassEst

    
    def adaClassifyWithStump(self,datToClass, classifierArr):    # 预测分类
        X = datToClass    # 测试数据集转为矩阵格式
        m = shape(X)[0]
        aggClassEst = zeros((m,1))
        for i in range(len(classifierArr)):
            classEst = self.stumpClassify(X, classifierArr[i]['dim'], classifierArr[i]['thresh'],classifierArr[i]['ineq'])     # 可以对比文章开头的图，其实就是那个公式
            aggClassEst += classifierArr[i]['alpha'] * classEst
        returnMat = zeros((m,1))
        minVal = aggClassEst.min()
        maxVal = aggClassEst.max()
        midVal = (minVal + maxVal) / 2
        returnMat[aggClassEst > midVal] = 1
        return returnMat

#--------------------------------------logistic---------------------------------
    def sigmoid(self,x):
        return 1/(1+exp(-x))

    def logisticClassify(self,testMat,weightArr):#testMat
        m = testMat.shape[0]
        testMat = append(testMat, ones((m,1)), axis = 1)
        predic = self.sigmoid(dot(testMat,weightArr.T))   #(m*n)*(n*1)->(m*1)
        predictedVals = ones((m,1)) #>0.5则为1，<0.5则为0
        predictedVals[predic < 0.5] = -1
        return predictedVals

    def trainLogistic(self,X,Y,D,epochs=200,learn_rate=0.006,alpha=6,delta=0.02):
        #D传进来就是一个m维列向量
        decent = 0
        Y = Y.reshape(-1,1) #m维行向量变成m维列向量
        m = X.shape[0]
        X = append(X, ones((m,1)), axis = 1) #增加一列全1
        n = X.shape[1]
        weight_arr = ones((1,n))/n
        for i in range(epochs):      
            Z = dot(X,weight_arr.T)  # m*n的矩阵与n*1的矩阵相乘，得到m维列向量，每个值为w1x1+w2x2+...+w57x57+b
            Y_hat = self.sigmoid(Z)
            grad = dot(X.T, (Y_hat - Y)*D)  #(n*m)*(m*1)->(n*1)
            grad_norm = linalg.norm(grad, ord = 2)   #求梯度的范数
            if (grad_norm <= delta):
                break
                
            else:
                weightArr = weightArr - learn_rate*grad.T #梯度下降,θ=θ−α∗(y_hat−y)∗x

        predic = self.sigmoid(dot(X,weightArr.T))   #(m*n)*(n*1)->(m*1)即m维列向量
        Y_predicted = zeros((m,1)) #>0.5则为1，<0.5则为0
        Y_predicted[predic > 0.5] = 1
        errArr = ones((m,1))#m维列向量
        errArr[Y_predicted==Y] = 0   # 预测值与实际值相同，误差置为0
        #print(type(errArr))
        #print(shape(D),shape(errArr))
        weightedError = dot(D.reshape(1,-1), errArr)     # D就是每个样本点的权值，随着迭代，它会变化，这段代码是误差率的公式 
        return weightArr,weightedError,Y_predicted

    def adaBoostTrainWithLogistic(self,X, classLabels,epoch=10):
        weakClassArr = []   # 保存每个基分类器的信息，存入列表
        m = shape(X)[0]
        D = ones((m,1))/m   #列向量
        aggClassEst = zeros((m,1))
        
        for i in range(epoch):
            bestWeights = {}
            weightArr,error,classEst = self.trainLogistic(X,classLabels,D)
            alpha = float(0.5 * log((1.0-error)/max(error,1e-16)))   # 对应公式 a = 0.5* (1-e)/e
            bestWeights['alpha'] = alpha
            bestWeights['weights'] = weightArr
            weakClassArr.append(bestWeights)  # 把每个基分类器存入列表
            expon =  alpha * ((classLabels.reshape(-1,1) != classEst)*2-1).reshape(-1,1)   # 预测结果与实际值相等则减小权重，否则增加
            D = D * exp(expon)       # 根据公式 w^m+1 = w^m (e^-a*y^i*G)/Z^m
            D = D/D.sum()         # 归一化
            aggClassEst += alpha * classEst   # 分类函数 f(x) = a1 * G1
        return weakClassArr,aggClassEst
    
    def adaClassifyWithLogistic(self,X,classifierArr):
        m = shape(X)[0]
        aggClassEst = zeros((m,1))
        #threshVal = 0
        for i in range(len(classifierArr)):
            aggClassEst += classifierArr[i]['alpha'] * self.logisticClassify(X,classifierArr[i]['weights'].reshape(1,-1))
            #threshVal += 0.5 * classifierArr[i]['alpha']
        returnMat = zeros((m,1))
        '''
        minVal = aggClassEst.min()
        maxVal = aggClassEst.max()
        midVal = (minVal + maxVal) / 2
        returnMat[aggClassEst > midVal] = 1
        '''
        returnMat[aggClassEst >= 0] = 1
        return returnMat

    def fit(self,dataName, targetName):
        time_start = time.time()
        X = loadtxt(dataName,float,delimiter=',')
        labelList = loadtxt(targetName,float,delimiter=',')
        if self.base == 0:   #基分类器为决策树桩
            minError = inf
            for epoch in [1, 5, 10, 100]:
                for i in range(10):
                    outputFileName = "./experiments/base%d_fold%d.csv" % (epoch, i+1)
                    trainMat, trainLabelList, selectedIndexList, testMat, testLabelList = self.kFoldSplit(X,labelList)              
                    weakClassArr, aggClassEst = self.adaBoostTrainWithStump(trainMat,trainLabelList,epoch)
                    ansList = self.adaClassifyWithStump(testMat,weakClassArr)   # 预测的分类结果，测试集我们用的是[0.7,1.7]测试集随便选
                    #print("结果是:",ansList)
                    ansMat = selectedIndexList.reshape(-1,1)
                    ansMat = append(ansMat,ansList.reshape(-1,1),axis = 1)
                    savetxt(outputFileName, ansMat, delimiter=',')
                    
                    corrList = zeros((ansList.shape[0], 1))
                    corrList[ansList != testLabelList.reshape(-1, 1)] = 1
                    errorRate = float(corrList.sum()/corrList.shape[0])
                    print(errorRate)
                    if(errorRate < minError):
                        minError = errorRate
                        self.classArr = weakClassArr
                        
            print(1-minError)
        elif self.base == 1:
            m,n = X.shape
            for i in range(n):
                mean = X[:,i].sum()/m     #求出第i个特征值的平均值
                stDeviation = std(X[:,i])     #求出第i个特征值的标准差
                X[:,i] = (X[:,i]-mean) / stDeviation    #对该特征值进行零均值规范化（归一化）
            minError = inf
            for epoch in [1,5,10,100]:
                totErrorRate = 0
                for i in range(10):
                    outputFileName = "./experiments/base%d_fold%d.csv" % (epoch, i+1)
                    trainMat, trainLabelList, selectedIndexList, testMat, testLabelList = self.kFoldSplit(X,labelList)              
                    weakClassArr, aggClassEst = self.adaBoostTrainWithLogistic(trainMat,trainLabelList,epoch)
                    ansList = self.adaClassifyWithLogistic(testMat,weakClassArr)   # 预测的分类结果，测试集我们用的是[0.7,1.7]测试集随便选
                    #print("结果是:",ansList)
                    print(epoch)
                    ansMat = selectedIndexList.reshape(-1,1)
                    ansMat = append(ansMat,ansList.reshape(-1,1),axis = 1)
                    savetxt(outputFileName, ansMat, delimiter=',')
                    corrList = zeros((ansList.shape[0], 1))
                    corrList[ansList != testLabelList.reshape(-1, 1)] = 1
                    errorRate = float(corrList.sum()/corrList.shape[0])
                    totErrorRate += errorRate / 10
                    if(errorRate < minError):
                        minError = errorRate
                        self.classArr = weakClassArr
                print(totErrorRate)
            print(1-minError)

    def predict(self, fileName):
        testMat = loadtxt(fileName, float, delimiter=',')
        if self.base == 0:
            ansMat = self.adaClassifyWithStump(testMat, self.classArr)
        elif self.base == 1:
            ansMat = self.adaClassifyWithLogistic(testMat, self.classArr)
        return ansMat

#=====================================没用====================================     
class LogisticClassifier:
    def __init__(self,learning_rate,epoch,delta):
        self.lr_rate = learning_rate #学习率
        self.epochs = epoch #迭代次数
        self.delta = delta
        self.w = None #每个特征值的权重系数

    

    def fit(self,X,labelList):
        decent = 0
        labelList = labelList.reshape(-1,1) #m维行向量变成m维列向量
        #print(labelList.shape)
        n = X.shape[1]
        self.w = zeros((1,n))  # 根据训练数据特征值的维数初始化参数w的维度以及形状
        for i in range(self.epochs):
            ## 计算W.T 点乘 x
            #print(i)        
            z = dot(X,self.w.T)  # m*n的矩阵与n*1的矩阵相乘，得到m维列向量
            h_theta = self.sigmoid(z)
            #print(h_theta.shape)
            error = h_theta - labelList  # y是一个n维列向量，计算误差 
            error_ = X*error   # m*n的矩阵与n*1的矩阵相乘，得到m维列向量
            # 计算梯度
            decent = error_.sum(axis=0)  ## 这个是梯度，并不是指的loss
                     
            if (abs(decent) <= self.delta).all():
                break
            else:
                self.w = self.w -self.lr_rate*decent #梯度下降,θ=θ−α∗(y_hat−y)∗x
        '''
        print('Weight: ')
        print(self.w[0][:-1])
        print('bias:')
        print(self.w[0][-1]) 
        '''
        return self.w

    def predict(self,X_test):
        m = X_test.shape[0]
        X_test = append(X_test,ones((m,1)),axis=1)
        predic = self.sigmoid(dot(X_test,self.w))
        result = asarray((predic >= 0.5).astype(int)) #>0.5则为1，<0.5则为0
        print('----------------------------')
        print('The predict label is :')
        print(result[0][0])
        return result

    def adaBoostTrainDS(X, classLabels,epoch=40):   # 迭代40次，直至误差满足要求，或达到40次迭代
        weakClassArr = []   # 保存每个基分类器的信息，存入列表
        m =shape(X)[0] 
        D = mat(ones((m,1))/m)
        aggClassEst = mat(zeros((m,1)))
        for i in range(epoch):
            bestStump,error,classEst = buildStump(X,classLabels,D)
            #print('D: ',D)
            alpha = float(0.5 * log((1.0-error)/max(error,1e-16)))   # 对应公式 a = 0.5* (1-e)/e
            bestStump['alpha']=alpha
            weakClassArr.append(bestStump)  # 把每个基分类器存入列表
            #print('classEst: ',classEst.T)
            #expon = multiply(-1 * alpha * mat(classLabels).T, classEst)   # multiply是对应元素相乘
            D = multiply(D,exp(expon))           # 根据公式 w^m+1 = w^m (e^-a*y^i*G)/Z^m
            D = D/D.sum()                 # 归一化
            aggClassEst += alpha * classEst      # 分类函数 f(x) = a1 * G1
            #print("aggClassEst: ",aggClassEst.T)
            aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T, ones((m,1)))   # 分错的矩阵
            errorRate = aggErrors.sum() /m   # 分错的个数除以总数，就是分类误差率
            #print('total error: ',errorRate)
            if errorRate <= 0.15:         # 误差率满足要求，则break退出
                break
        return weakClassArr,aggClassEst

    def adaClassify(datToClass, classifierArr):    # 预测分类
        X = mat(datToClass)    # 测试数据集转为矩阵格式
        m = shape(X)[0]
        aggClassEst = mat(zeros((m,1)))
        for i in range(len(classifierArr)):
            classEst = stumpClassify(X, classifierArr[i]['dim'], classifierArr[i]['thresh'],classifierArr[i]['ineq'])     # 可以对比文章开头的图，其实就是那个公式
            aggClassEst += classifierArr[i]['alpha']*classEst
            #print(aggClassEst)
        return sign(aggClassEst)


if __name__ =='__main__':                             # 运行函数
    time_start = time.time()
    test1 = Adaboost(base = 0)
    test1.fit('data.csv','targets.csv')
    time_end = time.time()
    print("运行时间：",time_end - time_start,'秒')