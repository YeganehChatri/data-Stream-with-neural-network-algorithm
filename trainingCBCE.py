from collections import deque
from scipy.spatial import distance
from numpy import array
from numpy import sum
from numpy import mean
from numpy import std
import json

classLabel=[]
ensemble={}
finalCF={}
potentialCF={}
prediction={}

b=0.9

recvQueue = deque([])

##create file for predicted test values  
f= open("prediction.txt","w+")
##read data 
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
	##extract classlabel of record
	tuple=recvQueue.popleft()
	
	
	def train(tuple):
		temp=tuple		
		del temp[0]

		##check for class evolution
		try:
			classLabel.index(tuple[0])
		except ValueError:
			classLabel.append(tuple[0])
			ensemble[tuple[0]]={'positive':[temp],'negative':[],'priorProbab':1}

		else:
		##check for new class or reccurence 2nd appearance
			if ensemble[tuple[0]]['priorProbab']>1:
				w=ensemble[key]['priorProbab']
				ensemble[key]['priorProbab']=1/w
		##check for class recurrence for the 1st time
			elif ensemble[tuple[0]]['priorProbab']==0:
				ensemble[key]['priorProbab']=1

			else:
		##updating existing class for +ve example
				ensemble[tuple[0]]['positive'].append(temp)
				w=ensemble[tuple[0]]['priorProbab']
				w=b*w+(1-b)
				ensemble[tuple[0]]['priorProbab']=w
		## updating existing classes for negative example
				for key in ensemble:
					if key is not tuple[0] and ensemble[key]['priorProbab']!=0 and ensemble[key]['priorProbab']<1:
						ensemble[key]['negative'].append(temp)
						w=ensemble[key]['priorProbab']
						w=b*w
						if w<b**100:
							w=0
						ensemble[key]['priorProbab']=w
					elif ensemble[key]['priorProbab']>=1:
						w=ensemble[key]['priorProbab']+1
						ensemble[key]['priorProbab']=w

		##microcluster formation

		##if no point exist for that class label
		if tuple[0] not in potentialCF:
			potentialCF[tuple[0]]=[[temp]]
			finalCF[tuple[0]]={}
			finalCf[tuple[0]]={
				'microCent':[],
				'linearSum':[0,0,0,0,0],
				'centroid':[0,0,0,0,0],
				'squareSum':[0,0,0,0,0],
				'radius':[0,0,0,0,0]
			}
	
		##if class label exist 
		else:
			##check for potential cluster where cluster exist
			k=0
			for i in range(0,len(potentialCF[tuple[0]])):
				for j in range(0,len(potentialCF[tuple[0]][i])):
					dist=distance.euclidean(temp,potentialCF[tuple[0]][i][j])
					if (dist<4.5):
						potentialCF[tuple[0]][i].append(temp)
						break
				k=i
				break
			## checking for potential conversion to final microcluster
			if len(potentialCF[tuple[0]][k])>=5:
				data_sum=sum(array(potentialCF[tuple[0]][k]), 0)
				centriod=array(data_sum)/5.0
				del potentialCF[tuple[0]][k]
				finalCf[tuple[0]]['microCent'].append(centroid)
				linear_sum = sum(array([finalCf[tuple[0]]['linearSum'],centroid]), 0)
				square = array(centroid) ** 2
				square_sum = sum(array([finalCf[tuple[0]]['squareSum'],square]),0)
				finalCf[tuple[0]]['centroid'] = array(finalCf[tuple[0]]['linearSum']) / float(len(finalCf[tuple[0]]['microCent']))
				finalCf[tuple[0]]['radius']=sum(array([finalCf[tuple[0]]['squareSum'],(-1)*array(finalCf[tuple[0]]['linearSum'])**2],0))
				
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


	def test(tuple):
		del(record[0])
		for key in ensemble:
			prob=calculateProbability(record,mean(ensemble[key]['positive']),stdev(ensemble[key]['positive']))*ensemble[key]['priorProbab']
			dist=distance.euclidean(record,finalCf[key]['centroid'])
			if(dist<finalCf[key]['radius']):
				prediction[key]=[prob,dist,'Y']
			else:
				prediction[key]=[prob,dist,'N']
			return prediction
	
	def predict(my_tuple):
		result=test(my_tuple)
		del(record[0])

		if my_tuple[0] is not '':
				train(my_tuple)
		else:
			predict(my_tuple) 
