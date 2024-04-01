#!/usr/bin/env python
# coding: utf-8

# In[4]:


#!/usr/bin/env python
# coding: utf-8

# In[7]:


import os


# In[8]:


filein ='../origin/unitigs_trim.Rtab'
outputfile = './unitigs.tped'

st = 'unitig'
fout = open(outputfile,'w')
file2 = open('unitigs.tfam','w')
index = -1
with open(filein) as file:
    for line in file:
        index = index+1
        if index ==0: 
            #fout.write(line) 
            ls = line.split()
            for l_ in ls[1:]:
                file2.write(l_+' '+l_+' 0 0 0 -9\n')

            continue
    
        line.lstrip() #removes any leading characters 
        #print(len(line))
        if len(line)==0:
            continue
        #print(line)
        # line begins with node_
        idx =5
        for idx in range(5,len(line)):
            if line[idx].isspace()==True: #Check if all the characters in the text are whitespaces:
                break
        for ind1 in range(len(line)):
            if line[ind1]=='_':
                break
        #while line[idx].isspace() == False :
        #    print(idx)
        #    idx = idx+1
        #    if idx  == len(line):
        #        break
        if idx == len(line):
            continue

        #print(idx)
        #ids = line[ind1+1:idx]   
        #print(ids)
        ln =  line[idx:]
        ln.lstrip()
        #print(ln)
        for l_ in ls[6:11]:
            ln = ln.replace('0','A A')
            ln = ln.replace('1','C C')
            fout.write('1 '+st+l_+' 0 21 '+ln) 
fout.close()
file2.close()   



# In[ ]:





# In[1]:





# In[ ]:




