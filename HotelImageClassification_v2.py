'''
This is a hotel image classification project.
Since running entire code on a single machine takes extravegant time , this version is created.
The code uses pickle to dump models, variable to create step wise execution.
'''



import cv2
import numpy as np
from collections import defaultdict
from sklearn.decomposition import PCA
import csv
import pickle
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
from sklearn import svm
from sklearn import preprocessing

class imageClssification:
    groundtruth=None
    inputDescForEachImage=None
    descStack=None
    represntation=None
    Y=None
    kmeans=None
    surf=None
    model=None
    pca=None
    #filepath='F:/masters/ML/HotelImageClassification/'
    filepath = 'F:/masters/ML/project/'
    def loadData(self):
        with open(self.filepath+'train.csv', 'rb') as csvfile:
            d1 = defaultdict(list)
            d2 = defaultdict(list)
            reader = csv.DictReader(csvfile)
            i=0
            count=0
            self.surf = cv2.SURF()
            #self.surf.upright = True
            self.surf.hessianThreshold = 700
            self.surf.extended = False
            isItFirstTimeLoad= True
            isItFirstTimeTempLoad=True
            for row in reader:
                #print "i:",row
                gray = cv2.imread(self.filepath+'train/'+row['id']+'.jpg', 0)
                #gray = cv2.imread('D:/masters/spring 16/ML/project/train/' + row['id'] + '.jpg', 0)
                #print 'D:/masters/spring 16/ML/project/train/' + row['id'] + '.jpg'
                if gray is not None:
                    if count == 20000:
                        break
                    d1[row['id']].append(row['col1'])
                    d1[row['id']].append(row['col2'])
                    d1[row['id']].append(row['col3'])
                    d1[row['id']].append(row['col4'])
                    d1[row['id']].append(row['col5'])
                    d1[row['id']].append(row['col6'])
                    d1[row['id']].append(row['col7'])
                    d1[row['id']].append(row['col8'])

                    count=count+1
                    kp, des = self.surf.detectAndCompute(gray, None)
                    if des is not None:
                        d2[row['id']]=des

                        if isItFirstTimeTempLoad:
                            tempStackOfDesc=np.array(des)
                            isItFirstTimeTempLoad=False
                        else:
                            temp = np.array(des)
                            tempStackOfDesc=np.vstack((tempStackOfDesc,temp))


                    if count%1000 == 0:
                        if isItFirstTimeLoad:
                            self.descStack =tempStackOfDesc
                            isItFirstTimeLoad=False

                            tempStackOfDesc = None
                            isItFirstTimeTempLoad = True
                        else:
                            self.descStack = np.vstack((self.descStack, tempStackOfDesc))
                            tempStackOfDesc=None
                            isItFirstTimeTempLoad = True
                            print count,'-------------loaded------------'

                    if count % 5000 == 0:
                        pickle.dump(self.descStack, open(self.filepath + 'descStack'+str(count)+'.dmp', 'wb'))
                        self.descStack=None
                        isItFirstTimeLoad=True

                        print count, '-------------dumped into descStack'+str(count)+'.dmp'+'------------'

            if tempStackOfDesc is not None:
                self.descStack = np.vstack((self.descStack, tempStackOfDesc))

            tempStackOfDesc=None

            print '..........image loading done...........'
            print 'count: ',count
            # Dump it to the file so it can be read later on
            #pickle.dump(self.descStack,open(self.filepath+'descStack.dmp','wb'))

            self.descStack=None

            pickle.dump(d1, open(self.filepath + 'csvData.dmp', 'wb'))
            d1=None
            pickle.dump(d2, open(self.filepath + 'disData.dmp', 'wb'))
            d2=None
            print '..........dumping done...........'
            #self.descStack=pickle.load(open(self.filepath+'descStack.dmp','rb'))
            #print '..........pickle loading done...........'

    def load_d1d2(self):
        with open(self.filepath + 'train.csv', 'rb') as csvfile:
            self.inputDescForEachImage = defaultdict(list)
            self.groundtruth = defaultdict(list)
            reader = csv.DictReader(csvfile)
            i = 0
            count = -1
            self.surf = cv2.SURF()
            self.surf.hessianThreshold = 700
            self.surf.extended = False
            for row in reader:

                gray = cv2.imread(self.filepath + 'train/' + row['id'] + '.jpg', 0)
                if gray is not None:
                    count = count + 1
                    if count <= 7500:
                        continue
                    if count == 10001:
                        break
                    self.groundtruth[row['id']].append(row['col1'])
                    self.groundtruth[row['id']].append(row['col2'])
                    self.groundtruth[row['id']].append(row['col3'])
                    self.groundtruth[row['id']].append(row['col4'])
                    self.groundtruth[row['id']].append(row['col5'])
                    self.groundtruth[row['id']].append(row['col6'])
                    self.groundtruth[row['id']].append(row['col7'])
                    self.groundtruth[row['id']].append(row['col8'])

                    print count
                    kp, des = self.surf.detectAndCompute(gray, None)
                    if des is not None:
                        self.inputDescForEachImage[row['id']] = des


                '''if count % 5000 == 0:
                    pickle.dump(d1, open(self.filepath + 'groundtruth'+ str(count) + '.dmp', 'wb'))
                    d1 = defaultdict(list)
                    print count, '-------------dumped into groundtruth' + str(count) + '.dmp' + '------------'

                    pickle.dump(d2, open(self.filepath + 'inputDescForEachImage' + str(count) + '.dmp', 'wb'))
                    d2 = defaultdict(list)
                    print count, '-------------dumped into inputDescForEachImage' + str(count) + '.dmp' + '------------'
                '''

        print '-----------loading d1d2 done--------------'

    def load_grndTruthDump(self):
        self.d1= pickle.load(open(self.filepath + 'groundtruth20000.dmp', 'rb'))
        print self.d1
        print '..........pickle loading done...........'
        self.d2 = pickle.load(open(self.filepath + 'inputDescForEachImage20000.dmp', 'rb'))
        print self.d2
        print '..........pickle loading done...........'

    def PCA_analysis(self):
        descStack1 = pickle.load(open(self.filepath+'descStack5000.dmp','rb'))
        descStack2 = pickle.load(open(self.filepath + 'descStack10000.dmp', 'rb'))
        descStack3 = pickle.load(open(self.filepath + 'descStack15000.dmp', 'rb'))
        #descStack4 = pickle.load(open(self.filepath + 'descStack20000.dmp', 'rb')) // this dump is kept for testing
        print '..........pickle loading done...........'

        self.descStack=np.array(descStack1)
        self.descStack = np.vstack((self.descStack, descStack2))
        self.descStack = np.vstack((self.descStack, descStack3))
        #self.descStack = np.vstack((self.descStack, descStack4)) // This dump is kept for testing
        print '..........descStack loading done...........'

        self.pca = PCA(n_components=15) # after seeing the plot decided on 15
        self.pca.fit(self.descStack)
        plt.bar(range(len(self.pca.explained_variance_)), self.pca.explained_variance_, align='center', alpha=0.5)
        plt.show()
        print '..........pca done...........'
        pickle.dump(self.pca, open(self.filepath + 'PCA_fit.dmp', 'wb'))
        print '..........pca dumping done...........'

    def runKmeans(self):
        #self.pca = PCA(n_components=15)
        descStack1 = pickle.load(open(self.filepath + 'descStack5000.dmp', 'rb'))
        descStack2 = pickle.load(open(self.filepath + 'descStack10000.dmp', 'rb'))
        descStack3 = pickle.load(open(self.filepath + 'descStack15000.dmp', 'rb'))
        # descStack4 = pickle.load(open(self.filepath + 'descStack20000.dmp', 'rb')) // this dump is kept for testing
        print '..........pickle loading done...........'

        self.descStack = np.array(descStack1)
        self.descStack = np.vstack((self.descStack, descStack2))
        self.descStack = np.vstack((self.descStack, descStack3))
        # self.descStack = np.vstack((self.descStack, descStack4)) // This dump is kept for testing
        print '..........descStack loading done...........'

        self.pca = pickle.load(open(self.filepath + 'PCA_fit.dmp', 'rb'))
        print '..........pca loading done...........'

        self.descStack=self.pca.transform(self.descStack)
        print '..........transform done...........'
        self.kmeans = KMeans(n_clusters=150, n_init=1, max_iter=100)
        self.kmeans.fit(self.descStack)
        print 'error: ',self.kmeans.inertia_
        print '..........Kmeans done...........'
        pickle.dump(self.kmeans, open(self.filepath + 'kmeans_fit.dmp', 'wb'))
        print '..........kmeans dumping done...........'

    def createBOFRepresentation(self):

        '''inputDescForEachImage1 = pickle.load(open(self.filepath + 'inputDescForEachImage5000.dmp', 'rb'))
        inputDescForEachImage2 = pickle.load(open(self.filepath + 'inputDescForEachImage10000.dmp', 'rb'))
        inputDescForEachImage3 = pickle.load(open(self.filepath + 'inputDescForEachImage15000.dmp', 'rb'))
        print '..........pickle inputDescForEachImage loading done...........'

        self.inputDescForEachImage = np.array(inputDescForEachImage1)
        self.inputDescForEachImage = np.vstack((self.inputDescForEachImage, inputDescForEachImage2))
        self.inputDescForEachImage = np.vstack((self.inputDescForEachImage, inputDescForEachImage3))
        print '..........inputDescForEachImage loading done...........'

        groundtruth1 = pickle.load(open(self.filepath + 'groundtruth5000.dmp', 'rb'))
        groundtruth2 = pickle.load(open(self.filepath + 'groundtruth10000.dmp', 'rb'))
        groundtruth3 = pickle.load(open(self.filepath + 'groundtruth15000.dmp', 'rb'))
        print '..........pickle groundtruth loading done...........'

        self.groundtruth = np.array(groundtruth1)
        self.groundtruth = np.vstack((self.groundtruth, groundtruth2))
        self.groundtruth = np.vstack((self.groundtruth, groundtruth3))
        print '..........groundtruth loading done...........'
        '''

        self.pca = pickle.load(open(self.filepath + 'PCA_fit.dmp', 'rb'))
        print '..........pca loading done...........'

        self.kmeans = pickle.load(open(self.filepath + 'kmeans_fit.dmp', 'rb'))
        print '..........kmeans loading done...........'

        isItFirstTimeTempLoad =True
        isItFirstTimeLoad = True
        count = 0
        X=None
        Y=None

        for key in self.inputDescForEachImage.keys():
            if self.inputDescForEachImage[key] is not None:

                rep = self.BOF(self.inputDescForEachImage[key], 150) # cluster size is 150
                z = np.array(self.groundtruth[key])
                ind = np.where(z == '1')
                count = count+1
                if isItFirstTimeTempLoad :
                    tempX = np.array(rep)
                    tempY = np.array(ind[0])
                    isItFirstTimeTempLoad =False
                else:
                    temp = np.array(rep)
                    tempX = np.vstack((tempX, temp))
                    tempY = np.append(tempY, ind[0])

            if count % 500 == 0:
                if isItFirstTimeLoad:
                    X = tempX
                    Y = tempY
                    tempX = None
                    tempY = None
                    isItFirstTimeLoad = False
                    isItFirstTimeTempLoad = True
                    print count, '-------------loaded------------'
                else:
                    X = np.vstack((X, tempX))
                    Y = np.append(Y, tempY)
                    tempX = None
                    tempY = None
                    isItFirstTimeTempLoad = True
                    print count, '-------------loaded------------'

            if count % 2500 == 0:
                pickle.dump(X, open(self.filepath + 'XStack10000.dmp', 'wb'))
                print count, '-------------dumped into XStack10000.dmp' + '------------'
                pickle.dump(Y, open(self.filepath + 'YStack10000.dmp', 'wb'))
                print count, '-------------dumped into YStack10000.dmp' + '------------'
                X = None
                Y = None
                isItFirstTimeLoad = True

        print '--------------------Representation done---------------------'

    def BOF(self, input, clusterSize):
        input = self.pca.transform(input)
        pre = self.kmeans.predict(input)
        x = plt.hist(pre, clusterSize, [0, clusterSize])
        return x[0]



    def pca_analysis2(self):
        XStack1 = pickle.load(open(self.filepath + 'XStack2500.dmp', 'rb'))
        XStack2 = pickle.load(open(self.filepath + 'XStack5000.dmp', 'rb'))
        XStack3 = pickle.load(open(self.filepath + 'XStack7500.dmp', 'rb'))
        XStack4 = pickle.load(open(self.filepath + 'XStack10000.dmp', 'rb'))
        XStack5 = pickle.load(open(self.filepath + 'XStack12500.dmp', 'rb'))
        XStack6 = pickle.load(open(self.filepath + 'XStack15000.dmp', 'rb'))
        XStack7 = pickle.load(open(self.filepath + 'XStack17500.dmp', 'rb'))
        XStack8 = pickle.load(open(self.filepath + 'XStack20000.dmp', 'rb'))
        XStack9 = pickle.load(open(self.filepath + 'XStack22500.dmp', 'rb'))
        print '..........pickle XStack loading done...........'

        XStack = np.array(XStack1)
        XStack = np.vstack((XStack, XStack2))
        XStack = np.vstack((XStack, XStack3))
        XStack = np.vstack((XStack, XStack4))
        XStack = np.vstack((XStack, XStack5))
        XStack = np.vstack((XStack, XStack6))
        XStack = np.vstack((XStack, XStack7))
        XStack = np.vstack((XStack, XStack8))
        XStack = np.vstack((XStack, XStack9))

        print '..........XStack loading done...........'

        self.pca = PCA(n_components=150)  # after seeing the plot decided on 15
        self.pca.fit(XStack)
        plt.bar(range(len(self.pca.explained_variance_)), self.pca.explained_variance_, align='center', alpha=0.5)
        plt.show()
        print '..........pca done...........'
        pickle.dump(self.pca, open(self.filepath + 'PCA_fit2.dmp', 'wb'))
        print '..........pca dumping done...........'


    def train(self):
        XStack1 = pickle.load(open(self.filepath + 'XStack2500.dmp', 'rb'))
        XStack2 = pickle.load(open(self.filepath + 'XStack5000.dmp', 'rb'))
        XStack3 = pickle.load(open(self.filepath + 'XStack7500.dmp', 'rb'))
        XStack4 = pickle.load(open(self.filepath + 'XStack10000.dmp', 'rb'))
        XStack5 = pickle.load(open(self.filepath + 'XStack12500.dmp', 'rb'))
        XStack6 = pickle.load(open(self.filepath + 'XStack15000.dmp', 'rb'))
        XStack7 = pickle.load(open(self.filepath + 'XStack17500.dmp', 'rb'))
        #XStack8 = pickle.load(open(self.filepath + 'XStack20000.dmp', 'rb'))
        #XStack9 = pickle.load(open(self.filepath + 'XStack22500.dmp', 'rb'))
        print '..........pickle XStack loading done...........'

        XStack = np.array(XStack1)
        XStack = np.vstack((XStack, XStack2))
        XStack = np.vstack((XStack, XStack3))
        XStack = np.vstack((XStack, XStack4))
        XStack = np.vstack((XStack, XStack5))
        XStack = np.vstack((XStack, XStack6))
        XStack = np.vstack((XStack, XStack7))
        #XStack = np.vstack((XStack, XStack8))
        #XStack = np.vstack((XStack, XStack9))

        print '..........XStack loading done...........'

        YStack1 = pickle.load(open(self.filepath + 'YStack2500.dmp', 'rb'))
        YStack2 = pickle.load(open(self.filepath + 'YStack5000.dmp', 'rb'))
        YStack3 = pickle.load(open(self.filepath + 'YStack7500.dmp', 'rb'))
        YStack4 = pickle.load(open(self.filepath + 'YStack10000.dmp', 'rb'))
        YStack5 = pickle.load(open(self.filepath + 'YStack12500.dmp', 'rb'))
        YStack6 = pickle.load(open(self.filepath + 'YStack15000.dmp', 'rb'))
        YStack7 = pickle.load(open(self.filepath + 'YStack17500.dmp', 'rb'))
        #YStack8 = pickle.load(open(self.filepath + 'YStack20000.dmp', 'rb'))
        #YStack9 = pickle.load(open(self.filepath + 'YStack22500.dmp', 'rb'))
        print '..........pickle YStack loading done...........'

        YStack = np.array(YStack1)
        YStack = np.vstack((YStack, YStack2))
        YStack = np.vstack((YStack, YStack3))
        YStack = np.vstack((YStack, YStack4))
        YStack = np.vstack((YStack, YStack5))
        YStack = np.vstack((YStack, YStack6))
        YStack = np.vstack((YStack, YStack7))
        #YStack = np.vstack((YStack, YStack8))
        #YStack = np.vstack((YStack, YStack9))

        print '..........YStack loading done...........'

        XStack = preprocessing.scale(XStack)
        YStack = YStack.transpose().flatten()

        X_train,X_test,Y_train,Y_test = cross_validation.train_test_split(XStack,YStack,train_size=0.9,test_size=0.1)
        #self.model = RandomForestClassifier(n_estimators=500).fit(X_train,Y_train)
        self.model= svm.SVC(kernel='rbf').fit(X_train,Y_train)

        print self.model.score(X_test,Y_test)
        #print self.model.n_classes_
        '''for i in range(0,1000):
            if YStack[i]==0:
                print YStack[i]
                plt.bar(range(len(XStack[i])), XStack[i], align='center', alpha=0.5)
                plt.show()
                print '--------------------------'
        '''
        print '--------------------------------------------------------------training done---------------------------------------------------------------------'
imgC=imageClssification()
#imgC.load_d1d2()
#imgC.load_grndTruthDump()
#imgC.PCA_analysis()
#imgC.runKmeans()
#imgC.loadData()
#imgC.createBOFRepresentation()
#imgC.pca_analysis2()
imgC.train()
