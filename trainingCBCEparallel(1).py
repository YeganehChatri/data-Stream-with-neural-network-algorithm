from mpi4py import MPI
from collections import deque
from numpy import array
from numpy import sum
from numpy import mean
from numpy import std
import json
import time

#process detail
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

##data structure for program
dataQueueSize=20
testQueueSize=5
classLabel={'current':{},'disappeared':{}} ##
disappearingClass=[] ##
recvQueue=deque([])
positive=[]
negative=[]
prior_probability=1
b=0.9
potential_cluster=[]
final_cluster={
	'microCent':[],
	'linearSum':[0,0,0,0,0],
	'centroid':[0,0,0,0,0],
	'squareSum':[0,0,0,0,0],
	'radius':[0,0,0,0,0]
}


##master node

##create file for predicted node

f= open("prediction.txt","w+")

## data read till eof of data file 
count=1
with open('data.csv', 'rb') as f:
	reader = csv.reader(f)
	for row in reader:
		if (i%20)==0:
			UpdateCBCE(recvQueue)
			recvQueue=[]
			i=i+1
			time.sleep(4)
		else: 
			print(row)
			recvQueue.append(row)
			i=i+1

def UpdateCBCE(recvQueue):
	if rank==0:
	## checking if a class has disappereaed the removing from current list and adding to disappeared list	
		if len(disappearingClass)!=0:
			temp=set(disappearingClass)
			for key, value in classLabel['current'].iteritems():
				if value in temp:
					classLabel['disappeared'][key]=value
					del(classLabel['current'][key])

	## send classlabel mapping to each node			
		messageTag=0	
		for key, value in classLabel['current'].iteritems():
			comm.send(classLabel,dest = value,tag=messageTag)
	
	## send record with classlabel to all nodes 
		while len(recvQueue)!=0:
			record=recvQueue.popleft()
			# label = tuple[0]
			
			##check for class evolution and recurrence
			if label not in classLabel['current'] and label not in classLabel['disappeared'] and label is not '':
				nodeID=len(classLabel['current'])+1
				classLabel['current'][label]=nodeID
			elif label not in classLabel['current'] and label is not '':
				classLabel['current'][label]=classLabel['disappeared'][label]
				del(classLabel['disappeared'][key])
			messageTag = messageTag + 1
			print("node",rank,"sending", data,"to slaves")
			for key, value in classLabel['current'].iteritems():
				comm.send(record,dest = value,tag=messageTag)

	
	comm.Barrier()

	##training node
	if rank in range(1,len(classLabel)):

	## create cluster
		def createCluster(record):
			for i in range(0,len(potential_cluster)):
				for j in range(0,len(potential_cluster[i])):
					dist=distance.euclidean(record,potential_cluster[i][j])
					if (dist<4.5):
						potential_cluster[i].append(record)
						break
				k=i
				break
			if len(potentialCF[k])>=5:
				data_sum=sum(array(potential_cluster[k]), 0)
				centriod = array(data_sum)/5.0
				del potential_cluster[k]
				final_cluster['microCent'].append(centroid)
				linear_sum=sum(array([final_cluster['linearSum'],centroid]),0)
				square=array(centroid)**2
				square_sum=sum(array([final_cluster['squareSum'],square]),0)
				final_cluster['centroid']=array(final_cluster['linearSum'])/float(len(final_cluster['microCent']))
				final_cluster['radius'] = final_cluster['squareSum'] - array(final_cluster['linearSum'])**2

	## trainig NB classifier of each class label
		def trainNb(record):
			if rank==classLabel['current'][record[0]]:
				del(record[0])
				createCluster(record)
				positive.append(record)
				if prior_probability>1:
					prior_probability=1/prior_probability
				prior_probability=b*prior_probability+(1-b)
	
			elif prior_probability>=1:
				prior_probability=prior_probability+1
			else:
				del(record[0])
				for i in range(len(negative)-len(positive)):
					del(negative[i])
					
				negative.append(record)
				w=b*prior_probability
				if w<b**100:
					w=0
					disappearingClass.append[rank]
				prior_probability=w

		def mean(numbers):
			return mean(array(numbers),axis=0)

		def stdev(numbers):
			return std(array(numbers),axis=0)

		def calculateProbability(x, mean, stdev):
			avg=array(mean)*(-1)
			numerator=math.pow(sum([x,avg],0),2)
			denominator=(2*math.pow(array(stdev),2))
			for i in range(len(numerator)):
				exponent = math.exp(-(numerator[i]/denominator[i]))
				prob_attribute=(1 / (math.sqrt(2*math.pi) * stdev[i])) * exponent
				prob *=prob_attribute
			return prob
			 

		def testNB(record):
			del(record[0])
			prob=calculateProbability(record,mean(positive),stdev(positive))*prior_probability
			dist=distance.euclidean(record,final_cluster['centroid'])
			if(dist<final_cluster['radius']):
				result[record]=[rank,prob,dist,'Y']
			else:
				result[record]=[rank,prob,dist,'N']
			return result

		## receive data from master and updated list of classlabel
		recv=deque([])
		classLabel=comm.recv(source=0,tag=0)	 
		for i in range(1,dataQueueSize):
			record=comm.recv(source=0, tag=i)
			recv.append(record)
		for i in range(len(recv)):
			record=recv.popleft()
			if record[0] is '':
				data=testNB(record)
				comm.send(data,dest = 0,tag=rank)
			else:
				trainNB(record)		
				comm.send(1,dest = 0,tag=rank)


	## each node updating disappearing class queue
	if prior_probability==0:
		disappearingClass.append[comm.bcast(rank,root=rank)]

	comm.Barrier()

	if rank==0:
		def predict(result):
			bestLabel, bestProb ,mindist= None, -1, 0
			for key, value in data.iteritems():
				if bestLabel is None or bestProb<value[0]:
					bestProb = value[0]
					bestLabel = key
				elif bestprob==value[0] and result[key][1]>result[bestlabel][1]:
					bestLabel = key
			for key, value in classLabel.iteritems():
				if value is bestLabel:
					bestLabel=key
			return bestLabel
			


	## receive prediction data from nodes  
		predict_result={}
		for key, value in classLabel['current'].iteritems():
			data=(comm.recv(source = value,tag=value))
			if data!=1:
				unpredictedRecord=data.keys()[0]
				for key, value in data.iteritems():
					if value[3] is not 'N':
						predict_result[value[0]]=[value[1],value[2]]
		json.dump(unpredictedRecord,f)
		if any(predict_result):
			json.dump(predict(predict_result),f)
			f.write("\n")
		else:
			
			f.write("novel class\n")
			

		
	
	
		
						
		

