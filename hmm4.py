import numpy as np
import csv

from copy import copy
import time

from sklearn.utils import shuffle
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pylab as plt

class HMM():
    
    def __init__(self):
        pass
        
    def customPrint(self,listOfArrays):
        i=1
        for arr in listOfArrays:
            print 'For Sequence ',i,'\n',arr
            i+=1
            
    def checkAccuracy(self,predicted,goldset):
        predicted=predicted.tolist()
        goldset=goldset.tolist()
        correct=0
        for i in range(0,len(predicted)):
            if goldset[i]==predicted[i]:
                correct+=1
        
        return (float(correct)/len(predicted))*100
    
    def loadData(self):
        #since we need to store the the mean and covariance matrix from the entire dataset, we need the entire dataset in one place for it
        # but since we need to calculate the transition probablities individually for the sequence , we need the datasets separately
        #1) Make 2 datasets, 
            #1.1)One containing all the points from all the sequences. Use this to calculate the P(y), mean and covariance matrix
            #1.2) For each sequence separately we calculate the transitions and then store them up
            
        self.pz=np.zeros((1,3)) #for 3 labelsOfNeighbours
        self.transitionTable=np.zeros((3,3))#here we store the respective counts for the corressponding state transitions
        self.means=np.zeros((3,2))# one row for each class  
        allDataX=[]
        allLabels=[]
        acc1=[]
        acc2=[]
        for i in range(1,6):
            datasetsXTrain=[]
            labelsyTrain=[]
            Xname="data/trainDataSeq"+str(i)+".csv"
            fx=open(Xname)
            xReader=csv.reader(fx)
           
            for row in xReader:
                row=[float(x) for x in row]
                datasetsXTrain.append(row)
                allDataX.append(row)
            
            fx.close()
            yname="data/trainLabelSeq"+str(i)+'.csv'
            fy=open(yname)
            
            yReader=csv.reader(fy)
            for row in yReader:
                row=[int(x) for x in row]
                #print 'row is ',row[0]
                labelsyTrain.append(row[0])
                allLabels.append(row[0])
            fy.close()
            datasetsXTrain=copy(np.asarray(datasetsXTrain))   
            labelsyTrain=copy(np.asarray(labelsyTrain)) 
            self.X_train=copy(datasetsXTrain)
            self.Y_train=copy(labelsyTrain)
            #from each individual dataset , find the respective counts of the transitions and store them in  TransitionTables , later normalize
            for i in range(0,self.transitionTable.shape[0]):
                for j in range(0,self.transitionTable.shape[0]):
                    self.transitionTable[i,j]+=self.countTransition(j+1,i+1)
                    
            #
            self.pz[0,self.Y_train[0]-1]+=1
            
        #now we have the P(Z) too. Now mean and covariance need to be calculated
                    
            
        #now we have the 2 datasets, one per sequence and one full
        #from the full one estimate the P(Z) and mean and the covairance.     
        #now normalize each row of the transition table
        self.normalize()
        allDataX=copy(np.asarray(allDataX))
        allLabels=copy(np.asarray(allLabels))
        #now the transition table is normalized, so now calculate P(Z) i.e. the starting probabilties from allData
        #now we need to normalize self.pz too
        
        x=float(sum(self.pz[0]))
        self.pz=self.pz/x
        s1=[1,2,3]
        for label in s1:
            rows=np.where(allLabels==label)[0]
            c=rows.shape[0]
            arr2=copy(allDataX[rows])
            self.means[label-1]=sum(arr2)/float(c)
        
        #now we have the means too, calculate the covariance matrix
        rows1=np.where(allLabels==1)[0]
        arr1=copy(allDataX[rows1])
        covMat1=copy(np.dot((arr1-self.means[0]).T,arr1-self.means[0]))
        
        rows2=np.where(allLabels==2)[0]
        arr2=copy(allDataX[rows2])
        covMat2=copy(np.dot((arr2-self.means[1]).T,arr2-self.means[1]))
        
        rows3=np.where(allLabels==3)[0]
        arr3=copy(allDataX[rows3])
        covMat3=copy(np.dot((arr3-self.means[2]).T,arr3-self.means[2]))
        
        self.covMat=copy((covMat1+covMat2+covMat3)/allLabels.shape[0])
        '''
        print 'The parameters for the HMM Model are\n'
        print 'Mean per class : (each row is a mean vector) \n',self.means,'\n'
        print 'the Covariance Matrix is \n',self.covMat,'\n'
        print 'The starting class priors are \n', self.pz,'\n'
        print 'The transition table is \n',self.transitionTable,'\n' '''
        #now we have the covariance matrix ad means and the transition probabilties and the starting probabilties
        #now testing 
        for i in range(1,6):
            datasetsXTest=[]
            labelsyTest=[]
            
            Xname="data/testDataSeq"+str(i)+".csv"
            fx=open(Xname)
            xReader=csv.reader(fx)
            X=[]
            y=[]
            for row in xReader:
                row=[float(x) for x in row]
                datasetsXTest.append(row)
            fx.close()
            yname="data/testLabelSeq"+str(i)+'.csv'
            fy=open(yname)
            y=[]
            yReader=csv.reader(fy)
            for row in yReader:
                row=[int(x) for x in row]
                #print 'row is ',row[0]
                labelsyTest.append(row[0])  
            fy.close()
            datasetsXTest=copy(np.asarray(datasetsXTest))  
            labelsyTest=copy(np.asarray(labelsyTest))
            self.X_test=copy(datasetsXTest)  
            self.Y_test=copy(labelsyTest)
            
           # print 'Monitoring Test Sequence : ',i,'\n'
            self.monitoring()
            acc=self.checkAccuracy(self.predictions,self.Y_test)
            acc1.append(acc)
            print 'Monitoring Accuracy of Test Sequence ',i , ' is ',acc
            #print 'Now doing Viterbi\n'
            self.viterbi()
            acc=self.checkAccuracy(self.predictions2,self.Y_test)
            #print 'Viterbi accuracy  for Test Sequence ',i,' is ',acc
            acc2.append(acc)
            fig=plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(datasetsXTest[:,0],datasetsXTest[:,1],labelsyTest,c=labelsyTest)
            plt.title('3d plot of Test Sequence '+str(i))
            plt.xlabel('X0')
            plt.ylabel('X1')
            plt.savefig('3d plot of Test Sequence '+str(i))
            plt.show()
            plt.scatter(datasetsXTest[:,0],datasetsXTest[:,1],c=labelsyTest)
            plt.title('2d plot of Test Sequence '+str(i))
            plt.xlabel('X0')
            plt.ylabel('X1')
            plt.savefig('2d plot of Test Sequence '+str(i))
            plt.show()
        print 'Mean accuarcy for monitoring is ',np.mean(acc1)
        #print 'Mean accuracy for Viterbi is ',np.mean(acc2)
        
        #ax = fig.add_subplot(111, projection='3d')
        
    def normalize(self):
        for i in range(0,3):
            x=sum(self.transitionTable[i])
            self.transitionTable[i]=copy(self.transitionTable[i]/float(x)) 
    

        
    def countTransition(self,label2,labelGiven):
        count=0
        for i in range(1,self.Y_train.shape[0]):
            if self.Y_train[i-1]==labelGiven and self.Y_train[i]==label2:
                count+=1

        return count
      
    def exponential(self,x,mu):
        #no need to pass the covariance matrix , since it would be same, 
        
        temp=-1*np.dot(x-mu,np.dot(np.linalg.inv(self.covMat),(x-mu).T))
        temp=temp/2
        return np.exp(temp)
            
    def monitoring(self):
        
        self.dpArray=copy(self.pz)
        s1=[1,2,3]
        #we initailize it with the starting probabilties , then keep on stacking up the probabilties SO THAT WE DO NOT HAVE TO CALCULATE them again and again
        self.predictions=[]
        for i in range(0,self.X_test.shape[0]):
            #although we are starting from 0, we are calculating P(y_1|X_1)
            #iterate over the labels  and get the maximum probability 
            valuesOverLabelsForCurrentY=[]
            for label in s1:
                tempSum=0
                for label2 in s1:
                    tempSum+=(self.dpArray[i,label2-1]*self.transitionTable[label2-1,label-1])
                p_y_equals_label_given_Rest=self.exponential(self.X_test[i],self.means[label-1])*tempSum
                valuesOverLabelsForCurrentY.append(p_y_equals_label_given_Rest)
            
            m1=max(valuesOverLabelsForCurrentY)
            index=valuesOverLabelsForCurrentY.index(m1)
            self.predictions.append(index+1)
            valuesOverLabelsForCurrentY=copy(np.asarray(valuesOverLabelsForCurrentY))
            valuesOverLabelsForCurrentY=valuesOverLabelsForCurrentY.reshape(1,-1)
            self.dpArray=copy(np.vstack((self.dpArray,valuesOverLabelsForCurrentY)))
          
              
        self.predictions=copy(np.asarray(self.predictions))      
            
        
        
    def viterbi(self):
        #implement the viterbi algorithm for 
        #got it 
        # This is an another classic DP problem
        self.dpArray2=np.zeros((1,3)) #for the three labels
        s1=[1,2,3]
        for label1 in s1:
            l=[]
            for label2 in s1:
                l.append(self.pz[0,label2-1]*self.transitionTable[label2-1,label1-1])
            m1=max(l)
            index=l.index(m1)
            self.dpArray2[0,label1-1]=m1
            
        #now we have made the first row of the dpArray2. Now we need to just make the rest of the classification
        self.predictions2=[]
        for i in range(0,self.X_test.shape[0]):
            if i==0:
                #i.e. we have the first example
                l=[]
                for label1 in s1:
                    temp=self.exponential(self.X_test[i],self.means[label1-1])  
                    l.append(temp*self.dpArray2[i,label1-1])
                
                m1=max(l)
                index=l.index(m1)
                self.predictions2.append(index+1)
                l=copy(np.asarray(l))
                l=copy(l.reshape(1,-1))
                self.dpArray2=copy(np.vstack((self.dpArray2,l)))
            else:
                l2=[]
                for label1 in s1:
                    l=[]
                    temp=self.exponential(self.X_test[i],self.means[label1-1])
                    for label2 in s1:
                        temp2=self.transitionTable[label2-1,label1-1]*self.dpArray2[i,label2-1]
                        l.append(temp2)
                        
                    m1=max(l)
                    l2.append(m1*temp)
                 
                m2=max(l2)
                index=l2.index(m2)
                self.predictions2.append(index+1)
                l2=copy(np.asarray(l2))
                l2=copy(l2.reshape(1,-1))
                self.dpArray2=copy(np.vstack((self.dpArray2,l2)))
                 
           
        self.predictions2=copy(np.asarray(self.predictions2))
                    
                        
                        
                    
                      
            
                
                
                        
        
    def train(self):
    
        s1=[1,2,3]
        print 's1 is ',s1
        #rowsc1=np.where(self.Y_train==1)[0]
        #print 'rows are ',rows
        self.pz=np.zeros((1,len(s1)))
        print 'Y_train is  ',self.Y_train.shape
        for label in s1:
            print 'label is ',label
            rows=np.where(self.Y_train==label)[0]
            print 'rows are ',len(rows)
            self.pz[0,label-1]=float(len(rows))/self.Y_train.shape[0]
            
            
        print 'self.pz',' is ',self.pz
        
        #now we have P(Z). Now we have to calculate p(x|z) and p(z_n|z_n-1), i.e the emission and the transition probabilties respectively
        self.transitionTable=np.zeros((len(s1),len(s1)))
        for i in range(0,self.transitionTable.shape[0]):
            for j in range(0,self.transitionTable.shape[0]):
                self.transitionTable[i,j]=self.countTransition(j+1,i+1)
                
        print 'the transition table is \n',self.transitionTable
        #now we need the emission probabilties
        
        self.means=np.zeros((len(s1),self.X_train.shape[1]))
        
        for label in s1:
            self.means[label-1]=copy(self.findMean(label))
        
        #now we have the mean for the datasets
        print 'the mean matrix is \n',self.means
        #now we find the variance for the datasets
        


        
        for label in s1:
            if label==1:
                self.covMatForLabel1=copy(self.findVariance(label))
            elif label==2:
                self.covMatForLabel2=copy(self.findVariance(label))
            else:
                self.covMatForLabel3=copy(self.findVariance(label))
            

        
        

        

ob1=HMM()
ob1.loadData()
#ob1.train()
