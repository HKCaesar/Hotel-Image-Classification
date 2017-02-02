import cv2
import numpy as np
from matplotlib import pyplot as plt
from collections import defaultdict
import csv
import os,glob
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier

from sklearn.decomposition import PCA

class imageClssification:
    csvData=None
    disData=None
    onlyDisc=None
    represntation=None
    Y=None
    kmeans=None
    surf=None
    model=None
    pca=None
    filepath='E:/masters/ML/project/'
    def loadData(self):
        with open(self.filepath+'train.csv', 'rb') as csvfile:
            d1 = defaultdict(list)
            d2 = defaultdict(list)
            reader = csv.DictReader(csvfile)
            i=0
            self.surf = cv2.SURF()
            #self.surf.upright = True
            self.surf.hessianThreshold = 2000
            self.surf.extended = False
            for row in reader:
                gray = cv2.imread(self.filepath+'train/'+row['id']+'.jpg', 0)
                if gray is not None:
                    d1[row['id']].append(row['col1'])
                    d1[row['id']].append(row['col2'])
                    d1[row['id']].append(row['col3'])
                    d1[row['id']].append(row['col4'])
                    d1[row['id']].append(row['col5'])
                    d1[row['id']].append(row['col6'])
                    d1[row['id']].append(row['col7'])
                    d1[row['id']].append(row['col8'])
                    kp, des = self.surf.detectAndCompute(gray, None)
                    if des is not None:
                        d2[row['id']]=des
                        print i
                        if i==0:
                            self.onlyDisc=np.array(des)
                            print self.onlyDisc.shape
                        else:
                            temp = np.array(des)
                            self.onlyDisc=np.vstack((self.onlyDisc,temp))
                i = i + 1
                if i == 20000:
                    break
            self.csvData = d1
            #print self.csvData.keys()
            self.disData = d2
            #print self.disData.values()[0].shape
            print 'Loading done-----------------------------------------------------------------------'

    def runKmeans(self,clusterSize):
        self.pca = PCA(n_components=15)
        print 'pca Done'
        self.onlyDisc=self.pca.fit_transform(self.onlyDisc)
        #print self.onlyDisc.shape
        self.kmeans = KMeans(n_clusters=clusterSize)
        self.kmeans.fit(self.onlyDisc)
        print 'Clustering done-----------------------------------------------------------------------'
        i=0
        for key in self.disData.keys():
            if self.disData[key] is not None:
                rep=self.BOF(self.disData[key],clusterSize)
                z=np.array(self.csvData[key])
                ind=np.where(z=='1')

                print i
                if i==0:
                    self.represntation = np.array(rep)
                    self.Y=np.array(ind[0])
                else:
                    temp=np.array(rep)
                    #print self.Y.shape
                    self.Y=np.append(self.Y,ind[0])
                    self.represntation = np.vstack((self.represntation,temp))
            i=i+1

        print 'Representation done-----------------------------------------------------------------------'

    def BOF(self,input,clusterSize):
        input=self.pca.transform(input)
        pre = self.kmeans.predict(input)
        x = plt.hist(pre, clusterSize, [0, clusterSize]);
        return x[0]

    def train(self):
        self.model = RandomForestClassifier(n_estimators=300)
        self.model.fit(self.represntation,self.Y.transpose().flatten())
        print '--------------------------------------------------------------training done---------------------------------------------------------------------'

    def predict(self,clusterSize):
        i=0
        with open(self.filepath+'test.csv', 'wb') as csvfile:
            outputwriter = csv.writer(csvfile,quoting=csv.QUOTE_NONE)
            for infile in glob.glob(os.path.join(self.filepath + 'test/', '*.*')):
                filename=os.path.split(infile)[1]
                gray = cv2.imread(self.filepath + 'test/' + filename, 0)
                if gray is not None:
                    kp, des = self.surf.detectAndCompute(gray, None)
                    if des is not None:
                        new = self.BOF(des, clusterSize)
                        #print np.array(new)
                        #out = self.model.predict(np.array(new))
                        out = self.model.predict_proba(np.array(new))
                        print filename, ' label:', out
                        arr=out.flatten()
                    else:
                        arr = np.zeros(8)
                        for k in range(0, 7):
                            arr[k] = 0.25
                        print filename
                else:
                    arr = np.zeros(8)
                    for k in range(0,7):
                        arr[k]=0.25
                    print filename

                id=os.path.splitext(filename)[0]
                outputwriter.writerow([id,arr[0],arr[1],arr[2],arr[3],arr[4],arr[5],arr[6],arr[7]])
                print i
                #if i==50:
                #    break
                i=i+1

        print 'prediction done-------------------------------------------------------------------------------'


imgC=imageClssification()
imgC.loadData()
imgC.runKmeans(15)
imgC.train()
imgC.predict(15)