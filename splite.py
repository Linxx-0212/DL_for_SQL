f=open('./data/xssed.csv','r')
f2=open('./data/sqlnew.csv','r')
data=f.readlines();
data=data[:2500]
ff=open('./data/sql_xss.csv','w')
for i in data:
    ff.write(str(i))
data=f2.readlines();
data=data[:2500]
for i in data:
    ff.write(str(i))