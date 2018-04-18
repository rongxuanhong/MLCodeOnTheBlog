from numpy import *
import matplotlib.pyplot as plt


def loadDataSet(fileName):
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        linArr = line.strip().split('\t')
        dataMat.append([float(linArr[0]), float(linArr[1])])
        labelMat.append(float(linArr[2]))
    return dataMat, labelMat


def kernelTrans(X, sampleX, kernelOp):
    """
    计算K(train_x,x_i)
    :param X:[n_samples, n_features] 保存训练样本的矩阵
    :param sampleX: [1,n] 某一样本矩阵
    :param kernelOp: 携带核信息的元组:参数一给定核的名称；后面参数为核函数可能需要的可选参数
    :return: K (numSamples,1)=shape(K)
    """
    m = shape(X)[0]  # 样本数
    K = mat(zeros((m, 1)))
    if kernelOp[0] == 'linear':  # 线性核
        K = X * sampleX.T
    elif kernelOp[0] == 'rbf':  # 高斯核
        sigma = kernelOp[1]
        if sigma == 0: sigma = 1
        for i in range(m):
            deltaRow = X[i, :] - sampleX
            K[i] = exp(deltaRow * deltaRow.T / (-2.0 * sigma ** 2))
    else:
        raise NameError('Not support kernel type! You can use linear or rbf!')
    return K


class SvmStruct:
    def __init__(self, dataMatIn, labelMat, C, toler, kernelOp):
        """
        初始化所有参数
        :param dataMatIn: 训练集矩阵
        :param labelMat: 训练集标签矩阵
        :param C: 惩罚参数
        :param toler: 误差的容忍度
        :param kernelOp: 存储核转换所需要的参数信息
        """
        self.train_x = dataMatIn
        self.train_y = labelMat
        self.C = C
        self.toler = toler
        self.numSamples = shape(dataMatIn)[0]  # 样本数
        self.alphas = mat(zeros((self.numSamples, 1)))  # 初始化待优化的一组alpha
        self.b = 0
        self.errorCache = mat(zeros((self.numSamples, 2)))  # 第1列为有效标志位(表示已经计算)，第2列为误差值
        self.K = mat(zeros((self.numSamples, self.numSamples)))
        # 计算出训练集 train_x 与每个样本X[i,:]的核函数转换值，并按列存储，那么共有 numSamples 列
        # 这样提取存储，方便查询使用，避免重复性计算，提高计算效率
        for i in range(self.numSamples):
            self.K[:, i] = kernelTrans(self.train_x, self.train_x[i, :], kernelOp)


def calcError(svm, k):
    """
    计算第k个样本的预测误差
    :param k:
    :return:
    """
    # 不使用核函数的版本
    # fxk = float(multiply(svm.alphas, svm.train_y).T * (svm.train_x * svm.train_x[k, :].T)) + svm.b
    fxk = float(multiply(svm.alphas, svm.train_y).T * svm.K[:, k] + svm.b)  # 使用核函数得出的预测值
    Ek = fxk - float(svm.train_y[k])
    return Ek


def selectJ(svm, i, Ei):
    """
    寻找第二个待优化的alpha,并具有最大步长
    :param i: 第一个alpha值的下标
    :param svm:
    :param Ei:第一个alpha值对应的Ei
    :return:
    """
    maxK = 0
    maxStep = 0
    Ej = 0
    validEcacheList = nonzero(svm.errorCache[:, 0].A)[0]  # 从误差缓存矩阵中 得到记录所有样本有效标志位的列表(注：存的是索引)
    if (len(validEcacheList)) > 1:  # 选择具有最大步长的 j
        for k in validEcacheList:
            if k == i:
                continue
            Ek = calcError(svm, k)
            step = abs(Ei - Ek)
            if (step > maxStep):  # 选择 Ej 与 Ei 相差最大的那个 j，即步长最大
                maxK = k
                maxStep = step
                Ej = Ek
        return maxK, Ej
    else:  # 第一次循环采用随机选择法
        l = list(range(svm.numSamples))
        # 排除掉已选的 i
        seq = l[:i] + l[i + 1:]
        j = random.choice(seq)
        Ej = calcError(svm, j)
    return j, Ej


def cliAlpha(alpha, L, H):
    """
    控制alpha在L到H范围内
    :param alpha: 待修正的alpha
    :param H: 上界
    :param L: 下界
    :return:
    """
    if alpha > H:
        alpha = H
    if alpha < L:
        alpha = L
    return alpha


def updateError(svm, k):
    """
    第k个样本的误差存入缓存矩阵，再选择第二个alpha值用到
    :param svm:
    :param k: 样本索引
    :return:
    """
    Ek = calcError(svm, k)
    svm.errorCache[k] = [1, Ek]


def innerL(svm, i):
    """
    :param i: 第一个alpha值的下标
    :param svm:
    :return: 返回是否选出了一对 alpha 值
    """
    Ei = calcError(svm, i)  # 计算第一个alpha值对应样本的预测误差

    # 接下来需要选择违反KKT条件最严重的那个alphas[i]
    # 满足KKT条件的三种情况
    # 1.yi*f(i)>=1 且 alpha=0,样本点落在最大间隔外(分类完全正确的那些样本)
    # 2.yi*f(i)==1 且 alpha<C,样本点刚好落在最大间隔边界上
    # 3.yi*f(i)<=1 且 alpha==C,样本点落在最大间隔内部
    # 情况2，3中的样本点也叫做支持向量
    # 违背KKT条件的三种情况(与上面相反)
    # 因为 y[i]*Ei = y[i]*f(i) - y[i]^2 = y[i]*f(i) - 1, 因此
    # 1.若yi*f(i)<0,则y[i]*f(i)<1,如果alpha<C，那么就违背KKT(alpha==C 才正确)
    # 2.若yi*f(i)>0,则y[i]*f(i)>1,如果alpha>0,那么就违背KKT(alpha==0才正确)
    # 3.若yi*f(i)==0,那么y[i]*f(i)==1,此时，仍满足KKT条件，无需进行优化

    if ((svm.train_y[i] * Ei < -svm.toler) and (svm.alphas[i] < svm.C) or (svm.train_y[i] * Ei > svm.toler) and (
                svm.alphas[i] > 0)):  # 选择违反KKT条件最严重的alpha[i]
        j, Ej = selectJ(svm, i, Ei)  # 选择第二个alpha值的下标以及得到其对应的样本的预测误差
        alphaIold = svm.alphas[i].copy()  # 记录更新前的alpha值
        alphaJold = svm.alphas[j].copy()  # 记录更新前的alpha值
        # 确定 alpha 值 的上下界
        if (svm.train_y[i] != svm.train_y[j]):
            L = max(0, alphaJold - alphaIold)
            H = min(svm.C, svm.C + alphaJold - alphaIold)
        else:
            L = max(0, alphaIold + alphaJold - svm.C)
            H = min(svm.C, alphaIold + alphaJold)
        if L == H:  return 0
        # 不使用核函数版本
        # X_i = svm.train_x[i, :]
        # X_j = svm.train_x[j, :]
        # eta = 2.0 * X_i * X_j.T - X_i * X_i.T - X_j * X_j.T
        # 使用核函数版本
        eta = svm.K[i, i] + svm.K[j, j] - 2.0 * svm.K[i, j]  # 计算eta=k_ii+k_jj-2*k_ij
        if eta <= 0: print("WARNING  eta<=0");return 0
        svm.alphas[j] += svm.train_y[j] * (Ei - Ej) / eta  # 计算出最优的alpha_j，也就是第二个alpha 值
        svm.alphas[j] = cliAlpha(svm.alphas[j], L, H)  # 得到修正范围后的 alpha_j
        if abs(svm.alphas[j] - alphaJold) < 0.00001:  # alpha_j 变化太小，直接返回
            updateError(svm, j)
            return 0

        svm.alphas[i] += svm.train_y[i] * svm.train_y[j] * (alphaJold - svm.alphas[j])  # 由 alpha_j 推出 alpha_i
        updateError(svm, i)  # 更新样本 i 的预测值误差
        # 不使用核函数版本
        # b1 = b - Ei - label_i * (alpha_i - alphaIold) * X_i * X_i.T - label_j * (alpha_j - alphaJold) * X_i * X_j.T
        # b2 = b - Ej - label_i * (alpha_i - alphaIold) * X_i * X_j.T - label_j * (alpha_j - alphaJold) * X_j * X_j.T
        # 使用核函数版本
        # 计算阈值
        b1 = - Ei - svm.train_y[i] * (svm.alphas[i] - alphaIold) * svm.K[i, i] - svm.train_y[j] * (
            svm.alphas[j] - alphaJold) * svm.K[i, j] + svm.b
        b2 = - Ej - svm.train_y[i] * (svm.alphas[i] - alphaIold) * svm.K[i, j] - svm.train_y[j] * (
            svm.alphas[j] - alphaJold) * svm.K[j, j] + svm.b

        if (0 < svm.alphas[i]) and (svm.alphas[i] < svm.C):  # alpha_i 不在边界上，b1有效
            svm.b = b1
        elif (0 < svm.alphas[j]) and (svm.alphas[j] < svm.C):  # alpha_j 不在边界上，b2有效
            svm.b = b2
        else:  # alpha_j、alpha_j 都在边界上,阈值取中点
            svm.b = (b1 + b2) / 2
        updateError(svm, j)
        updateError(svm, i)
        return 1  # 一对alphas值已改变
    else:
        return 0


def smoP(dataSet, classLabels, C, toler, maxIter, KTup=('linear', 1.0)):
    svm = SvmStruct(mat(dataSet), mat(classLabels).T, C, toler, KTup)
    iter = 0
    entireSet = True  # 是否遍历所有alpha
    alphaPairsChanged = 0

    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:  # 对整个训练集遍历
            for i in range(svm.numSamples):
                alphaPairsChanged += innerL(svm, i)
            print('---iter:%d entire set, alpha pairs changed:%d' % (iter, alphaPairsChanged))
        else:  # 对非边界上的alpha遍历(即约束在0<alpha<C内的样本点)
            nonBoundIs = nonzero((svm.alphas.A > 0) * (svm.alphas.A < svm.C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(svm, i)
            print('---iter:%d non boundary, alpha pairs changed:%d' % (iter, alphaPairsChanged))
        iter += 1
        if entireSet:
            entireSet = False
        elif (alphaPairsChanged == 0):
            entireSet = True
    return svm


def calcWs(alphas, dataArr, labelArr):
    """
    计算W
    :param alphas: 大部分为0，非0的alphas对应的样本为支持向量
    :param dataArr:
    :param classLabels:
    :return:
    """
    X = mat(dataArr)
    labelMat = mat(labelArr).T
    m, n = shape(X)
    w = zeros((n, 1))
    for i in range(m):  ## alphas[i]=0的无贡献
        w += multiply(alphas[i] * labelMat[i], X[i, :].T)
    return w


def plotSVM():
    dataMat, labelMat = loadDataSet('testSet.txt')
    svm = smoP(dataMat, labelMat, 0.6, 0.001, 50)

    classified_pts = {'+1': [], '-1': []}
    for point, label in zip(dataMat, labelMat):
        if label == 1.0:
            classified_pts['+1'].append(point)
        else:
            classified_pts['-1'].append(point)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for label, pts in classified_pts.items():
        pts = array(pts)
        ax.scatter(pts[:, 0], pts[:, 1], label=label)

    supportVectorsIndex = nonzero(svm.alphas.A > 0)[0]
    for i in supportVectorsIndex:
        plt.plot(svm.train_x[i, 0], svm.train_x[i, 1], 'oy')

    w = calcWs(svm.alphas, dataMat, labelMat)

    x1 = min(array(dataMat)[:, 0])
    x2 = max(array(dataMat)[:, 0])

    a1, a2 = w
    y1, y2 = (-float(svm.b) - a1 * x1) / a2, (-float(svm.b) - a1 * x2) / a2
    ax.plot([x1, x2], [y1, y2])
    plt.show()

plotSVM()


def testSVMWithLinearKernel():
    dataArr, labelArr = loadDataSet('testSet.txt')
    svm = smoP(dataArr, labelArr, 0.6, 0.001, 50)
    dataMat = mat(dataArr)
    labelMat = mat(labelArr).T
    svInd = nonzero(svm.alphas.A > 0)[0]
    sVs = dataMat[svInd]
    labelSv = labelMat[svInd]
    print("there are %d Support Vector" % shape(sVs)[0])
    m, n = shape(dataMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs, dataMat[i, :], ('linear', 1.0))
        predict = kernelEval.T * multiply(labelSv, svm.alphas[svInd]) + svm.b
        if sign(predict) != sign(labelArr[i]): errorCount += 1
    print("the training error rate is : %f" % (errorCount / m))


# testSVMWithLinearKernel()


# dataMat, train_y = loadDataSet('testSet.txt')
# b, alphas = smoP(dataMat, train_y, 0.6, 0.001, 50)
# ws = calcWs(alphas, dataMat, train_y)
# errorCount = 0
# for i in range(shape(dataMat)[0]):
#     a = mat(dataMat)[i] * mat(ws) + b
#     if sign(a) != train_y[i]:
#         errorCount += 1
# print("the training error rate is : %f" % (errorCount / shape(dataMat)[0]))


# print(a)  # 预测的值 a>0 ==1 a<0 -1


# print(train_y[0])

def testRbf(k1=1.3):
    dataArr, labelArr = loadDataSet('testSetRBF.txt')
    svm = smoP(dataArr, labelArr, 200, 0.0001, 100, ('rbf', k1))
    dataMat = mat(dataArr)
    labelMat = mat(labelArr).T
    svInd = nonzero(svm.alphas.A > 0)[0]
    sVs = dataMat[svInd]
    labelSv = labelMat[svInd]
    print("there are %d Support Vector" % shape(sVs)[0])
    m, n = shape(dataMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs, dataMat[i, :], ('rbf', k1))
        predict = kernelEval.T * multiply(labelSv, svm.alphas[svInd]) + svm.b
        if sign(predict) != sign(labelArr[i]): errorCount += 1
    print("the training error rate is : %f" % (errorCount / m))

# testRbf()
