#coding=utf-8
import nltk
import re
from urllib.parse import unquote
import tensorflow as tf
import keras.backend.tensorflow_backend as ktf
def GeneSeg(payload):
    payload=payload.lower()
    payload=unquote(unquote(payload))
    payload,num=re.subn(r'\d+',"0",payload)
    payload,num=re.subn(r'(http|https)://[a-zA-Z0-9\.@&/#!#\?]+', "http://u", payload)
    #分词
    #print (u"decoding result："+str(payload))
    r = '''
        (?x)[\w\.]+?\(
        |\)
        |"\w+?"
        |'\w+?'
        |http://\w
        |<\w+
        |\w+=
        |<>
        |[\w\.]+
        |</\w+>
        |<\w+>
        |>
    '''
    return nltk.regexp_tokenize(payload, r)
def init_session():
    gpu_options=tf.GPUOptions(allow_growth=True)
    ktf.set_session(tf.Session())