import numpy as np

'''
机器学习大作业：分别实现以对数几率回归和决策树桩为基分类器的 AdaBoost 算法
姓名：丁兆睿
班级：计卓2001
学号：U202015292
指导教师：张腾
完成时间：2022.06.25
'''
class Adaboost:
    classifier_arr = None   # 用于保存基分类器组合
    index_partition = None  # 十折交叉验证时用于保存分组的索引值
    def __init__(self,base):
        self.base = base

    # 加载文件-----------------------------------------------------------------------------------------------------------
    '''
    # 输入：数据和标记的文件
    # 输出：处理缺失值后的数据矩阵和标记向量
    '''
    def load_file(self, data_name, targets_name):
        X = np.genfromtxt(data_name,float,delimiter=',')    # genfromtxt可以将缺失值读为np.nan，而loadtxt会报错
        Y = np.genfromtxt(targets_name,float,delimiter=',')
        X = np.nan_to_num(X, nan=0)     # 进行缺失值预处理，将缺失值位置填充为0
        Y = np.nan_to_num(Y, nan=0)
        return X, Y

    # k折交叉验证，此处为10折---------------------------------------------------------------------------------------------
    '''
    输入：整个数据矩阵、标记数组、交叉验证轮数
    返回：训练用数据，训练用标记，挑选出的用于验证的样本索引，验证用数据，验证用标记
    '''
    def k_fold_split(self, X, Y, num, k=10):
        m = X.shape[0]
        whole_index_list = [i for i in range(m)]
        left_index_list = whole_index_list
        self.index_partition = np.zeros((10,int(np.floor(m/k))))   # 将数据分为10组，这个矩阵每一行存储的就是一组样本的所有索引
        for i in range(k):
            selected_index_list = np.random.choice(left_index_list, size = int(np.floor(m/k)), replace = False)    # 随机取出剩余索引中的一份，并将其从剩余索引中删除
            left_index_list = list(set(left_index_list) - set(selected_index_list))
            self.index_partition[i] = selected_index_list
        self.index_partition = self.index_partition.astype('int64')     # 索引值必须为int或bool型
        test_index_list = self.index_partition[num]     # 当前取第num组作为测试数据，记录这些索引的值，以便与标记值一一对应
        X_test = X[test_index_list]     # 通过索引构成测试用的数据矩阵
        test_label_list = Y[test_index_list]        # 通过索引构成测试用的数据矩阵标记向量
        train_index_list = list(set(whole_index_list)-set(test_index_list))     # 将测试样本的索引删除，剩余的就是训练样本索引
        X_train = X[train_index_list]   # 通过索引构成训练用的数据矩阵
        Y_train = Y[train_index_list]   # 通过索引构成训练用的标记
        return X_train, Y_train, test_index_list, X_test, test_label_list

    # 决策树桩基分类器的建立---------------------------------------------------------------------------------------------
    
    def stump_classify(self,X,dimension,thresh_val,classify_method):
        '''
        决策树桩分类函数，通过阈值和分类方式将数据分为0和1，用于建立决策树桩时寻找阈值时，得到当前阈值和分类方式的分类结果
        输入：X为数据矩阵，dimension为当前特征值为第几维，thresh_val为当前阈值，classify_method为分类方式，有两种
        返回：预测结果
        '''
        m = X.shape[0]
        Y_pred = np.ones((m,1))   # 预测结果，m维列向量
        if classify_method == 'less':        # 表示分类方式，对于小于阈值的样本点赋值为0
            Y_pred[X[:,dimension] < thresh_val] = 0.0
        elif classify_method == 'geq':  # 大于阈值的样本点赋值为0
            Y_pred[X[:,dimension] >= thresh_val] = 0.0
        return  Y_pred            # 返回基分类器的分类结果,m维列向量

    def train_stump(self,X,Y,D):
        '''
        决策树桩建立函数，得到最佳的决策树桩信息，包括特征，阈值和分类方式，整合成一组key-value返回
        输入：X为数据矩阵，Y为标记向量，D为样本权重向量
        返回：最佳决策树桩，对应的最小加权错误率和
        '''
        Y = Y.reshape(-1,1)#转化为列向量
        m,n = X.shape
        #numStemp = 10
        best_stump = {}
        Y_pred_best = np.zeros((m,1))
        min_error = np.inf
        for i in range(n):                  # 遍历特征值的n维
            step_size = 300
            sorted_eigenvalues = np.sort(X[:,i])
            left = 0
            right = left + step_size - 1
            while right < m:
                thresh_val = (sorted_eigenvalues[left] + sorted_eigenvalues[right]) / 2     # 左右端点取平均作为阈值
                for classify_method in ['less', 'geq']:   # 两种分类方式：小于     
                    errors = np.ones((m,1))#m维列向量
                    Y_pred = self.stump_classify(X,i,thresh_val,classify_method)  # Y_pred为预测结果
                    errors[Y_pred==Y] =0   # 预测值与实际值相同，误差置为0
                    weighted_error = np.dot(D.reshape(1,-1), errors)     # 对错误样本乘以样本权重加和
                    if weighted_error < min_error:     # 选出分类误差最小的基分类器
                        best_stump['dimension'] = i          # 特征
                        best_stump['threshold'] = thresh_val    # 阈值
                        best_stump['method'] = classify_method        # 分类方式
                        Y_pred_best = Y_pred   # 更新分类的结果
                        min_error = weighted_error     # 更新最小误差
                left += step_size
                right += step_size
        return best_stump,min_error,Y_pred_best
    
    # 基于决策树桩的adaboost算法实现--------------------------------------------------------------------------------------

    def stump_classify_for_adaboost(self,X,dimen,thresh_val,classify_method):
        '''
        决策树桩分类函数，通过阈值和分类方式将数据分为-1和1，用于在Adaboost算法中最后得到预测结果时作为h(x)
        输入：X为数据矩阵，dimension为当前特征值为第几维，thresh_val为当前阈值，classify_method为分类方式，有两种
        返回：预测结果
        '''
        m = np.shape(X)[0]
        Y_pred = np.ones((m,1))   # 创造一个 样本数×1 维的array数组
        if classify_method == 'less':        # lt表示less than，表示分类方式，对于小于等于阈值的样本点赋值为-1
            Y_pred[X[:,dimen] < thresh_val] = -1
        elif classify_method == 'geq': 
            Y_pred[X[:,dimen] >= thresh_val] = -1
        return  Y_pred
    
    def adaBoost_train_with_stump(self,X, Y, epoch=10):   # 迭代10次
        '''
        基于决策树桩的adaboost训练函数，通过迭代epoch次得到epoch个基分类器以及它们的权重
        输入：X为数据矩阵，Y为标记值向量，epoch为迭代次数，即基分类器个数
        返回：基分类器列表
        '''
        m = X.shape[0]
        Y = Y.reshape(-1,1)     # 行向量->列向量
        base_classifier_arr = []   # 基分类器列表
        D = np.ones((m,1))/m    # 样本权重初始化为1/m
        for i in range(epoch):
            best_stump,error,Y_pred = self.train_stump(X,Y,D)
            alpha = float(0.5 * np.log((1.0-error)/max(error,1e-16)))   # α = 0.5* ln (1-e)/e
            best_stump['alpha']=alpha
            base_classifier_arr.append(best_stump)  
            power =  alpha * ((Y != Y_pred)*2-1).reshape(-1,1)   # 预测结果与实际值相等则减小权重，否则增加。括号中是将预测错误指数变成1,正确的指数变为-1
            D = D * np.exp(power)      
            D = D/D.sum()         # 归一化
        return base_classifier_arr

    
    def adaboost_classify_with_stump(self,X, classifier_arr):
        '''
        adaboost分类函数，通过adaboost算法得到的基分类器列表预测给定数据的标记值
        输入：X为数据特征值矩阵，classifier_arr为基分类器列表（数组）
        返回：预测值向量
        '''
        m = np.shape(X)[0]
        Y_est_sum = np.zeros((m,1))
        for i in range(len(classifier_arr)):
            Y_est = self. stump_classify_for_adaboost(X, classifier_arr[i]['dimension'], classifier_arr[i]['threshold'],classifier_arr[i]['method'])
            Y_est_sum += classifier_arr[i]['alpha'] * Y_est
        Y_pred = np.zeros((m,1))
        Y_pred[Y_est_sum >= 0] = 1
        return Y_pred

    # logistic回归基分类器的建立--------------------------------------------------------------------------------------------

    def sigmoid(self,x):
            return 1.0 / (1 + np.exp(-x))     # 当x为接近0的负数时，原函数会溢出，因此对x<0的情况单独处理
        
    def logistic_classify(self,X_test,weight_arr):
        '''
        对率回归分类函数，通过阈值和分类方式将数据分为0和1，用于建立对率回归模型时，得到分类结果
        输入：X为数据矩阵，weight_arr为w和b构成的向量
        返回：预测结果
        '''
        m = X_test.shape[0]
        Y_est = self.sigmoid(np.dot(X_test,weight_arr.T))   #(m*n)*(n*1)->(m*1)
        Y_pred = np.ones((m,1)) #>0.5则为1，<0.5则为0
        Y_pred[Y_est < 0.5] = 0
        return Y_pred

    def train_logistic(self,X,Y,D,epochs=200,learn_rate=0.005,delta=0.002):
        '''
        对率回归建立函数，得到拟合后的w和b值
        输入：X为数据矩阵，Y为标记向量，D为样本权重向量，epoch为迭代次数，learn_rate为学习率，delta为临界范数（可接受的梯度范数）
        返回：迭代后的w与b构成的向量，对应的加权错误率和预测结果
        '''
        #D传进来就是一个m维列向量
        Y = Y.reshape(-1,1) #m维行向量变成m维列向量
        m = X.shape[0]
        X = np.append(X, np.ones((m,1)), axis = 1) #增加一列1
        n = X.shape[1]
        weight_arr = np.ones((1,n))/n
        for i in range(epochs):      
            Z = np.dot(X,weight_arr.T)  # m*n的矩阵与n*1的矩阵相乘，得到m维列向量，每个值为w1x1+w2x2+...+w57x57+b
            Y_hat = self.sigmoid(Z)
            grad = np.dot(X.T, (Y_hat - Y)*D)  # (n*m)*(m*1)->(n*1)
            grad_norm = np.linalg.norm(grad, ord = 2)   # 求梯度的范数
            if (grad_norm <= delta):
                break
            weight_arr = weight_arr - learn_rate*grad.T     # 梯度下降

        Y_pred = self.logistic_classify(X, weight_arr)
        error_arr = np.ones((m,1))      # m维列向量
        error_arr[Y_pred==Y] = 0    # 预测值与实际值相同，误差置为0
        weighted_error = np.dot(D.reshape(1,-1), error_arr)     # 加权错误率
        return weight_arr,weighted_error,Y_pred
    
    # 基于对率回归的adaboost算法实现-------------------------------------------------------------------------------------

    def logistic_classify_for_adaboost(self,X_test,weight_arr):#X_test
        '''
        对率回归分类函数，通过阈值和分类方式将数据分为-1和1，用于在adaboost算法中最后得到预测结果时作为h(x)
        输入：X为数据矩阵，weight_arr为w和b构成的向量
        返回：预测结果
        '''
        m = X_test.shape[0]
        X_test = np.append(X_test, np.ones((m,1)), axis = 1)
        Y_est = self.sigmoid(np.dot(X_test,weight_arr.T))   # (m*n)*(n*1)->(m*1)
        Y_pred = np.ones((m,1))     # >0.5则为1，<0.5则为-1
        Y_pred[Y_est < 0.5] = -1
        return Y_pred
    
    def adaboost_train_with_logistic(self, X, Y, epoch=10):
        '''
        基于对率回归的adaboost训练函数，通过迭代epoch次得到epoch个基分类器以及它们的权重
        输入：X为数据矩阵，Y为标记值向量，epoch为迭代次数，即基分类器个数
        返回：基分类器列表
        '''
        base_classifier_arr = []   # 基分类器列表
        m = X.shape[0]
        D = np.ones((m,1))/m   # 列向量
        Y = Y.reshape(-1,1)
        for i in range(epoch):
            best_weights = {}
            weight_arr,error,Y_pred = self.train_logistic(X,Y,D)
            alpha = float(0.5 * np.log((1.0-error)/max(error,1e-16)))
            best_weights['alpha'] = alpha
            best_weights['weights'] = weight_arr
            base_classifier_arr.append(best_weights)  # 将基分类器存入列表
            power =  alpha * ((Y != Y_pred)*2-1).reshape(-1,1)   # 括号中是将预测错误指数变成1,正确的指数变为-1
            D = D * np.exp(power)
            D = D/D.sum()         # 归一化
        return base_classifier_arr
    
    def adaboost_classify_with_logistic(self,X,classifier_arr):
        '''
        adaboost分类函数，通过adaboost算法得到的基分类器列表预测给定数据的标记值
        输入：X为数据特征值矩阵，classifier_arr为基分类器列表（数组）
        返回：预测值向量
        '''
        m = X.shape[0]
        Y_est_sum = np.zeros((m,1))
        for i in range(len(classifier_arr)):
            Y_est_sum += classifier_arr[i]['alpha'] * self.logistic_classify_for_adaboost(X,classifier_arr[i]['weights'].reshape(1,-1))
        Y_pred = np.zeros((m,1))
        Y_pred[Y_est_sum >= 0] = 1
        return Y_pred

    # fit----------------------------------------------------------------------------------------------------------

    def fit(self,data_name, target_name):
        '''
        fit函数，通过文件中的数据进行adaboost模型构造和训练，最后得到十折交叉验证的预测结果，并保存其中正确率最高的基分类器组合。
        输入：数据的文件路径
        输出：十折交叉验证结果，输出到./experiments文件夹下的文件中
        '''
        X,Y=self.load_file(data_name, target_name)
        
        if self.base == 0:
            print("正在训练模型，这可能需要一分钟左右的时间……")
            m,n = X.shape
            for i in range(n):
                mean = np.mean(X[:,i])     # 求出第i个特征值的平均值
                st_deviation = np.std(X[:,i])     # 求出第i个特征值的标准差
                X[:,i] = (X[:,i]-mean) / st_deviation    # 对该特征值进行零均值规范化（归一化）
            min_error = np.inf
            for epoch in [1,5,10,100]:
                print("正在进行基分类器类型为[对数几率回归]，基分类器个数为[",epoch,"]的十折交叉验证")
                for i in range(10):
                    output_file_name = "./experiments/base%d_fold%d.csv" % (epoch, i+1)
                    X_train, Y_train, selected_index_list, X_test, Y_test = self.k_fold_split(X,Y,i)              
                    classifier_arr = self.adaboost_train_with_logistic(X_train,Y_train,epoch)
                    Y_pred = self.adaboost_classify_with_logistic(X_test,classifier_arr)   # 预测的分类结果
                    #print("结果是:",Y_pred)
                    ans_mat = selected_index_list.reshape(-1,1)     # ans_mat共有两列，第一列是索引，第二列是标记值，相当于一个索引到标记值的映射
                    ans_mat = np.append(ans_mat,Y_pred.reshape(-1,1),axis = 1)
                    np.savetxt(output_file_name, ans_mat, delimiter=',')      # 将结果输出至文件

                    error_arr = np.zeros((Y_pred.shape[0], 1))
                    error_arr[Y_pred != Y_test.reshape(-1, 1)] = 1
                    error_rate = float(error_arr.sum()/error_arr.shape[0])
                    if(error_rate < min_error):
                        min_error = error_rate
                        self.classifier_arr = classifier_arr

        elif self.base == 1:   #基分类器为决策树桩
            print("正在训练模型，这可能需要1.5分钟左右的时间……")
            min_error = np.inf
            for epoch in [1, 5, 10, 100]:
                
                print("正在进行基分类器类型为[决策树桩]，基分类器个数为[",epoch,"]的十折交叉验证")
                for i in range(10):
                    output_file_name = "./experiments/base%d_fold%d.csv" % (epoch, i+1)
                    X_train, Y_train, selected_index_list, X_test, Y_test = self.k_fold_split(X,Y,i)              
                    classifier_arr= self.adaBoost_train_with_stump(X_train, Y_train,epoch)
                    Y_pred = self.adaboost_classify_with_stump(X_test, classifier_arr)  
                    ans_mat = selected_index_list.reshape(-1,1)
                    ans_mat = np.append(ans_mat,Y_pred.reshape(-1,1),axis = 1)
                    np.savetxt(output_file_name, ans_mat, delimiter=',')
                    error_arr = np.zeros((Y_pred.shape[0], 1))
                    error_arr[Y_pred != Y_test.reshape(-1, 1)] = 1
                    error_rate = float(error_arr.sum()/error_arr.shape[0])
                    if(error_rate < min_error):
                        min_error = error_rate
                        self.classifier_arr = classifier_arr
        else:
            print("传入base时出现错误，请检查base是否为0或1")
        print("模型训练完毕，十折交叉验证结果已输出至文件")

    # predict-----------------------------------------------------------------------------------------------------------

    def predict(self, fileName):
        '''
        predict函数，对文件中所给的数据特征值，预测标记的结果
        输入：文件路径
        输出：预测标记值列表，为一个列向量
        '''
        X_test = np.genfromtxt(fileName, float, delimiter=',')
        X_test = np.nan_to_num(X_test, nan=0)
        if self.base == 0:      # 对率回归
            m,n = X_test.shape
            for i in range(n):      # 对率回归时，需要先对数据进行归一化处理
                mean = np.mean(X_test[:,i])     # 求出第i个特征值的平均值
                st_deviation = np.std(X_test[:,i])     # 求出第i个特征值的标准差
                X_test[:,i] = (X_test[:,i]-mean) / st_deviation    # 对该特征值进行零均值规范化（归一化）
            Y_pred = self.adaboost_classify_with_logistic(X_test, self.classifier_arr)
        elif self.base == 1:    # 决策树桩
            Y_pred = self.adaboost_classify_with_stump(X_test, self.classifier_arr)
        return Y_pred

if __name__ == '__main__':
    test1 = Adaboost(base = 0)      # 0为对率回归，1为决策树桩
    test1.fit('data.csv','targets.csv')