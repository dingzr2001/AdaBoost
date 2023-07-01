from numpy import *
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.font_manager import *
 
def loadDataSet():                # 加载测试数据
    dataList = [[1,1,1,1,1,1],[1,1,1,1,1,1],[1,1,1,1,1,1],[1,1,1,1,1,1],[0,0,0,0,0,0]]
    labelList = [0,0,0,0,1]
    dataMat = mat(dataList)
    return dataMat, labelList      # 数据集返回的是矩阵类型，标签返回的是列表类型
 
def stumpClassify(dataMat,dimen,threshVal,threshIneq): # dimen:第dimen列，也就是第几个特征, threshVal:是阈值  threshIneq：标志
    retArray = ones((shape(dataMat)[0],1))   # 创造一个 样本数×1 维的array数组
    if threshIneq == 'lt':        # lt表示less than，表示分类方式，对于小于等于阈值的样本点赋值为-1
        retArray[dataMat[:,dimen] <= threshVal] = 0.0
    else:  # 我们确定一个阈值后，有两种分法，一种是小于这个阈值的是正类，大于这个值的是负类，
           #第二种分法是小于这个值的是负类，大于这个值的是正类，所以才会有这里的if 和else
        retArray[dataMat[:,dimen] > threshVal] = 0.0
    return  retArray            # 返回的是一个基分类器的分类好的array数组
 
def buildStump(dataArr,classLabels,D):
    dataMat = mat(dataArr)
    labelMat = mat(classLabels).T
    m,n = shape(dataMat)
    numStemp = 100
    bestStump = {}
    bestClassEst = mat(zeros((m,1)))
    minError = inf                      # 无穷
    for i in range(n):                  # 遍历特征
        rangeMin = dataMat[:,i].min()    # 检查到该特征的最小值
        rangeMax = dataMat[:,i].max()
        print(type(dataMat[:,i]))
        print((dataMat[:,i]).tolist)
        print(type(dataMat[:,i].A))
        sortedValues = dataMat[:,i].A.sort()
        print(sortedValues)
        print(type(sortedValues))
        stepSize = (rangeMax - rangeMin)/numStemp  # 寻找阈值的步长是最大减最小除以10,你也可以按自己的意愿设置步长公式
        a = 0
        b = 9
        while b < 3680:
            for inequal in ['lt', 'gt']:   # 因为确定一个阈值后，可以有两种分类方式
                threshVal = (sortedValues[a]+sortedValues[b])/2
                predictedVals = stumpClassify(dataMat,i,threshVal,inequal)  # 确定一个阈值后，计算它的分类结果，predictedVals就是基分类器的预测结果，是一个m×1的array数组
                errArr = mat(ones((m,1)))
                errArr[predictedVals==labelMat] =0   # 预测值与实际值相同，误差置为0
                weightedEroor = D.T*errArr     # D就是每个样本点的权值，随着迭代，它会变化，这段代码是误差率的公式
                print(type(weightedEroor))
                if weightedEroor<minError:     # 选出分类误差最小的基分类器
                    minError=weightedEroor     # 保存分类器的分类误差
                    bestClassEst = predictedVals.copy()   # 保存分类器分类的结果
                    bestStump['dim']=i          # 保存那个分类器的选择的特征
                    bestStump['thresh']=threshVal    # 保存分类器选择的阈值
                    bestStump['ineq']=inequal        # 保存分类器选择的分类方式
            a+=10
            b+=10
    return bestStump,minError,bestClassEst
 
def adaBoostTrainDS(dataMat, classLabels,numIt=40):   # 迭代40次，直至误差满足要求，或达到40次迭代
    weakClassArr = []   # 保存每个基分类器的信息，存入列表
    m =shape(dataMat)[0]
    D = mat(ones((m,1))/m)
    aggClassEst = mat(zeros((m,1)))
    for i in range(numIt):
        bestStump,error,classEst = buildStump(dataMat,classLabels,D)
        #print('D: ',D)
        alpha = float(0.5 * log((1.0-error)/max(error,1e-16)))   # 对应公式 a = 0.5* (1-e)/e
        bestStump['alpha']=alpha
        weakClassArr.append(bestStump)  # 把每个基分类器存入列表
        #print('classEst: ',classEst.T)
        expon = alpha * ((mat(classLabels).T!=classEst)*2-1)   # multiply是对应元素相乘
        D = multiply(D,exp(expon))           # 根据公式 w^m+1 = w^m (e^-a*y^i*G)/Z^m
        D = D/D.sum()                 # 归一化
        aggClassEst += alpha * classEst      # 分类函数 f(x) = a1 * G1
        #print("aggClassEst: ",aggClassEst.T)
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T, ones((m,1)))   # 分错的矩阵
        errorRate = aggErrors.sum() /m   # 分错的个数除以总数，就是分类误差率
        #print('total error: ',errorRate)
        if errorRate == 0.0:         # 误差率满足要求，则break退出
            break
    return weakClassArr,aggClassEst
 
def adaClassify(datToClass, classifierArr):    # 预测分类
    dataMat = mat(datToClass)    # 测试数据集转为矩阵格式
    m = shape(dataMat)[0]
    aggClassEst = mat(zeros((m,1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMat, classifierArr[i]['dim'], classifierArr[i]['thresh'],classifierArr[i]['ineq'])     # 可以对比文章开头的图，其实就是那个公式
        aggClassEst += classifierArr[i]['alpha']*classEst
        #print(aggClassEst)
    return sign(aggClassEst)

 
if __name__ =='__main__':                             # 运行函数
    dataMat, labelList = loadDataSet()                     # 加载数据集
    weakClassArr, aggClassEst = adaBoostTrainDS(dataMat,labelList)
    print('weakClassArr',weakClassArr)
    print('aggClassEst',aggClassEst)
    classify_result =adaClassify([1,1,1,1,1],weakClassArr)   # 预测的分类结果，测试集我们用的是[0.7,1.7]测试集随便选
    print("结果是:",classify_result) 