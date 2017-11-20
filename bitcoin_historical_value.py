from exchanges.coindesk import CoinDesk as cd
import csv

#Open/Create a file to append data
csvFile = open('C:\\Users\\Marco\\Desktop\\bitcoin3.csv', 'a')
#Use csv Writer
csvWriter = csv.writer(csvFile, delimiter='\t' )

start_date = '2017-01-01'
end_date = '2017-11-18'
values = cd.get_historical_data_as_dict(start=start_date, end=end_date)
for date in values:
    csvWriter.writerow([date, values[date]])


