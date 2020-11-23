import json

import urllib

try:
    import urllib.request as req
except ImportError:
    import urllib as req

from pprint import pprint 
import time
from datetime import datetime

host = 'Avell'
my_token = '933860814:AAFxSdEFgcC3sSoQn1P39hv9clvctfl_sPs'
default_chat_id = -345852348

def getChats(token=my_token):
    result = req.urlopen("https://api.telegram.org/bot{}/getUpdates".format(token)).read()
    messages = json.loads(result.decode('utf-8'))['result']
    pprint(result)
    talks = {}
    for data in messages:
        #pprint(data)
        chat = data['message']['chat']
        talks[chat['id']] = "{} {} (@{})".format(chat['first_name'],chat['last_name'],chat['username'])
        pprint(talks)

def send(msg, token=my_token, chat_id=default_chat_id, print_=False):
    data = { "chat_id": chat_id, "text": msg }
    try:
        sendData = urllib.parse.urlencode(data).encode('ascii')
        if print_:
            pprint(msg)
    except Exception as e:
        print(e)
        #import urllib
        sendData = urllib.parse.urlencode(data)

    # Send a message to a chat room (chat room ID retrieved from getUpdates)
    result = req.urlopen("https://api.telegram.org/bot{}/sendMessage".format(token), sendData).read()

def send_message_trainning(host='Localhost', algoritmhs='Teste', model='default', start_train=None):
    current_time = time.time()
    dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    duration = None
    if start_train != None:
        duration = round((current_time - start_train)/60, 4)
    msg_ = '------------------------\n{}\n{} - {}\n{}\nDuration: {} minutes\n------------------------'.format(host, algoritmhs, model, dt_string, duration)
    print(msg_)
    send(msg_) 


def send_train_message_start(host='Localhost', algoritmhs='Teste', model='default'):
    dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    msg_ = '-------START TRAINING-------\nHOST: {}\n{} - {}\n{}\n------------------------'.format(host, algoritmhs, model, dt_string)
    send(msg_) 
    return time.time()   

def send_train_message_end(host='Localhost', algoritmhs='Teste', model='default', start_train=None):
    end_time = time.time()
    dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    duration = 0
    if start_train != None:
        duration = str(round((end_time - start_train)/3600, 4))
    msg_ = '-------END TRAINING-------\nHOST: {}\n{} - {}\n{}\nDuration: {} hours\n------------------------'.format(host, algoritmhs, model, dt_string, duration)
    send(msg_) 
    return end_time     
    
def send_message_error(host='Localhost', algoritmhs='Teste', model='default', start_train = None, error_message=""):
    end_time = time.time()
    dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    duration = ""
    if start_train != None:
        duration = str(round((end_time - start_train)/60, 4))
    msg_ = '----------ERROR--------\nHOST: {}\n{} - {}\n{}\nDuration: {} minutes\nError: {}\n------------------------'.format(host, algoritmhs, model, dt_string, duration, error_message)
    send(msg_) 

def setChatId(id):
    default_chat_id = id