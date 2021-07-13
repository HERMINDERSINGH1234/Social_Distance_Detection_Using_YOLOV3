# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 19:13:43 2020

@author: venkata Sreeram
"""


from flask import Flask

app=Flask(__name__)

@app.route('/')
def hello():
    return "Hey Hi everyone This was my First Flask Application !!"

if __name__=='__main__':
    app.run(debug=False)
    
    
    
