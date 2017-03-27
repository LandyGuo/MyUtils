#coding=utf-8
import sys
import random
import argparse



if __name__=="__main__":
	parser = argparse.ArgumentParser() #获取参数过滤器
	parser.add_argument("num",type=int,help="the number needs to be sampled")#位置参数1(根据位置来取值)

	args = parser.parse_args() 
	capacity = args.num

	reservior = [None]*capacity

	for line,content in enumerate(sys.stdin):
		content = content.strip()
		if line<capacity:
			reservior[line] = content
		else:
			j = random.randint(0, line)
			if j<capacity:
				reservior[j] = content

	for content in reservior:
		print content
