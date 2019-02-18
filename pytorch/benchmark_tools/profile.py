#! /usr/bin/python
import re
import pdb
import sys
import getopt

p1=re.compile('.*Running\s+operator\s+(.*)\((\S+)\).*')
p2=re.compile('.*This\s+operator\s+iteration\s+took\s+(\S+)\s+ms\s+to\s+complete.*')
p3=re.compile('.*Exit\s+after\s+running\s+(\S+)\s+iterations.*')
p4=re.compile('.*Output\s+shape:.*,\s+computing\s+in.*seconds,\s+processing.*')
p5=re.compile('.*INFO\s+test_engine.py:\s+.*\s+(\S+)\s+:\s+(\S+)')
p7=re.compile('.*INFO\s+infer_simple.py:\s+.*\s+(\S+\s+\S+):\s+(\S+)s')
p6=re.compile('.*INFO\s+infer_simple.py:\s+.*\s+\|\s+(\S+):\s+(\S+)s')


iteration=5000
mode='sameop'
filename='old/style_log'
try:
  opts, args = getopt.getopt(sys.argv[1:],"hi:o:n:m:",["ifile=","ofile=","iter=","mode="])
except getopt.GetoptError:
  print 'use -h for help'
  sys.exit(2)
for opt, arg in opts:
  if opt == '-h':
    print 'python profile.py -i <inputfile> -o <outputfile> -m mode'
    print ' '
    print 'for example: convert2caffe2.py -i input -o output -m mode'
    print 'input:  the input file'
    print 'output: the output file'
    print 'mode: all  or sameop '
    sys.exit()

  elif opt in ("-i", "--ifile"):
    filename = arg
  elif opt in ("-o", "--ofile"):
    output_file = arg
  elif opt in ("-n", "--iter"):
    iteration = int(arg)
  elif opt in ("-m", "--mode"):
    mode = arg




f_log=open(filename)
tag=0
sameop=0
block=0
op={}
eachop={}
eopnames=[]
totaltime=0
part={}
for i in f_log.readlines():
    #if 'Start running performanc' in i or :
        #pdb.set_trace()
    tag=1
    if tag == 0:
        continue
    else:
	#if 'enter profile log' in i:
        if 'Running net' in i and 'mask_net' not in i:
            eopnames=[]
	#    print "into new loop"
	#else if p4.match(i) != None:
        #    print "leave loop"
	    
	else:
            y5 = p5.match(i)
            if y5 !=None:
                partname=y5.group(1)
                partvalue=y5.group(2)
                if partname in part:
                    part[partname] += float(partvalue)
                else:
                    part[partname] = float(partvalue)
            y6 = p6.match(i)
            if y6 !=None:
                partname=y6.group(1)
                partvalue=y6.group(2)
                if partname in part:
                    part[partname] += float(partvalue)
                else:
                    part[partname] = float(partvalue)
            else:
                y7 = p7.match(i)
                if y7 !=None:
                    partname=y7.group(1)
                    partvalue=y7.group(2)
                    if partname in part:
                        part[partname] += float(partvalue)
                    else:
                        part[partname] = float(partvalue)
	    y1 = p1.match(i)
	    if y1!=None and sameop==0:
            	sameop=1
                if y1.group(1) !='':
                    npos=y1.group(1).index(':')
                    if npos >= 0:
                        extraname=y1.group(1)[0:npos-1]
                    else: 
                        extraname=y1.group(1)
            	    opname=extraname + y1.group(2)
            	    eopname=extraname + y1.group(2)
                else:
                    opname=y1.group(2)
                    eopname=y1.group(2)
		if not eopnames:
		    eopnames.append("start")
		    
            	for j in range(len(eopnames)):
                    if eopname in eopnames:
		        eopname = opname + str(j+2)
	            else:
		    	eopnames.append(eopname)
			break            
            elif y1!=None:
	    	print 'duplicate op line ', i
            else:
	    	y2 = p2.match(i)
            	if y2!=None and sameop==1:
                    sameop=0
                    optime=y2.group(1)
		    totaltime+=float(optime)
		    if mode == 'all':
		    	if eopname in eachop:
                    	    eachop[eopname]=float(eachop[eopname])+float(optime)
                    	else:
                    	    eachop[eopname]=float(optime)
                    if opname in op:
                    	op[opname]=float(op[opname])+float(optime)
                    else:
		    	op[opname]=float(optime)
            	elif y2!=None:
                    print 'can not find op before line ', i 
            	else:
		    y3=p3.match(i)
		    if y3!=None:
		    	iteration=y3.group(1)
		    	break
		    else:
		        continue
f_log.close()
print "iter = ", iteration

print part

print op
f_out=open(output_file,'w')
if mode == 'all':
    f_out.write("each op time list below" + "\n")
    #items=eachop.items()
    #items.sort()
    #for key, value in items:
    #	f_out.write('{0} : {1}'.format(key, float(value)/float(iteration)) + '\n')

    for i in eopnames[1:]:
	if i in eachop:
	    f_out.write('{0} , {1}'.format(i, float(eachop[i])/float(iteration)) + '\n')
	else:
	    print "no key find in dict, key= ", i
	    f_out.close()
	    exit 
    

f_out.write("sum of kinds of op time list below" + "\n")

oitems=op.items()
oitems.sort()
for key, value in oitems:
    f_out.write('{0} , {1}'.format(key, float(value)/float(iteration)) + '\n')

f_out.write("total time, {}".format(totaltime) + "\n")

f_out.write("part time list below" + "\n")

	     
for key, value in part.items():
    f_out.write('{0} , {1}'.format(key, float(value)*1000/float(iteration)) + '\n')        
	    
f_out.close()

