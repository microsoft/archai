'''
all utilities for enron dataset
'''

import numpy as np
import pandas as pd
import email
import datetime
from dateutil import parser

# reference: https://www.kaggle.com/ankurrezo/data-cleaning-and-transformation-enron-email
def get_field(field, messages):
    column = []
    for message in messages:
        e = email.message_from_string(message)
        column.append(e.get(field))
    return column
def change_type(dates):
    column = []
    for date in dates:
        column.append(parser.parse(date).strftime("%d-%m-%Y %H:%M:%S"))
    return column
df = pd.read_csv("/home/t-gjawahar/object_dir/enron-data/emails.csv") #, nrows=10)
df['date'] = get_field("Date", df['message'])
df['date'] = change_type(df['date'])
#print(df.loc[0]['date'])
#print(df.loc[19]['date'])
#print(df)
df[['date']] = df[['date']].apply(pd.to_datetime)
df.sort_values(by="date",  inplace=True)
#print(df)
#message = df.loc[100]['message']
#e = email.message_from_string(message)
#print(e.get_payload())
#print(e.get('date'))
print(df.shape[0], df.shape[1])
num_records = df.shape[0]
num_val = 5000
num_test = 5000
num_train = num_records - num_val - num_test
w_train = open("/home/t-gjawahar/object_dir/enron-char/wiki.train.tokens", "w")
w_val = open("/home/t-gjawahar/object_dir/enron-char/wiki.valid.tokens", "w")
w_test = open("/home/t-gjawahar/object_dir/enron-char/wiki.test.tokens", "w")
for i in range(num_records):
    message = df.loc[i]['message']
    e = email.message_from_string(message)
    res = ""
    for line in e.get_payload().split("\n"):
        if len(line.strip()) == 0:
            continue
        res += " %s \n"%(line)
    res += " \n"
    if i < num_train:
        w_train.write(res)
    elif i < num_train + num_val:
        w_val.write(res)
    else:
        w_test.write(res)
w_train.close()
w_val.close()
w_test.close()
