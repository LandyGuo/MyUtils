#coding=utf-8
import numpy as np
import argparse
import codecs

from sklearn.cross_validation import train_test_split

def parse_file(file):
	x,y = [],[]
	with codecs.open(file,'r','utf-8') as f:
		for line in f:
			cx, cy = line.split()
			cy = cy.strip()
			x.append(cx)
			y.append(cy)
	return x,y


if __name__=="__main__":
	parser = argparse.ArgumentParser() #获取参数过滤器
	parser.add_argument("indexfile",type=str,help="indexfile to split for train and test,X and y should be ‘\t’ separated")#位置参数1(根据位置来取值)
	parser.add_argument("-test_ratio","--ratio",type=float,default=0.2,help="factor of test data")
	parser.add_argument("-otr","--trfile",type=str,default="tr.txt",help="output train file name")
	parser.add_argument("-ote","--tefile",type=str,default="te.txt",help="output test file name")
	
	args = parser.parse_args() #实际从命令行中提取参数的过程
	file = args.indexfile
	X,Y = parse_file(file)
	X_train, X_test, y_train, y_test = train_test_split(
							X, Y, test_size=args.ratio, random_state=12)
	
	#collect data and labels
	o_train,o_test = [],[]
	for x,y in zip(X_train,y_train):
		o_train.append(x+'\t'+y+'\n')
	for x,y in zip(X_test,y_test):
		o_test.append(x+'\t'+y+'\n')
	#write results
	o_tr,o_te = args.trfile,args.tefile
	codecs.open(o_tr,'w+','utf-8').writelines(o_train)
	codecs.open(o_te,'w+','utf-8').writelines(o_test)
	
