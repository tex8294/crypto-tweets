import csv
import numpy as np

def compare(date1, date2):
    from datetime import datetime
    from datetime import timedelta
    date1 = datetime.strptime( date1, '%Y-%m-%d %H:%M');
    date2 = datetime.strptime( date2, '%Y-%m-%d %H:%M');
    return date1 < date2
    
def time_decrease(date):
    from datetime import datetime
    from datetime import timedelta
    date = datetime.strptime( date, '%Y-%m-%d');
    delta = timedelta(days=1);
    new_date = date-delta;
    new_date = date-delta;
    final_date = datetime.isoformat(new_date)[0:10];
    return final_date

def convert(date):
    from datetime import datetime
    from datetime import timedelta
    import time
    new_date = time.mktime(datetime.strptime(date, '%Y-%m-%d %H:%M').timetuple())
    return new_date

# Open/Create a file to append data
csvFile = open('C:\\Users\\Marco\\Desktop\\puta.csv', 'a')
    
#Use csv Writer
csvWriter = csv.writer(csvFile, delimiter='\t' )


with open('C:\\Users\\Marco\\Desktop\\tweets.csv') as f:
    r = csv.reader(f, delimiter=';')
    r = list(r)
    
with open('C:\\Users\\Marco\\Desktop\\val.csv') as f:
    s = csv.reader(f, delimiter=';')
    s = list(s)

            
xp=np.zeros(len(s))
yp=np.zeros(len(s))
cont = 0
for row in s:
    xp[cont]=convert(row[0])
    yp[cont]=float(row[1])
    cont = cont + 1

x=np.zeros(len(r))
cont = 0
for row in r:
    x[cont]=convert(row[2])
    cont = cont + 1

y=np.zeros(len(r))
y=np.interp(x,xp,yp)

cont = 0
for row in r:
   csvWriter.writerow([row[0],row[2],row[3],y[cont]])
   cont = cont+1;
        


