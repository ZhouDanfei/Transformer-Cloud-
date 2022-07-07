#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flask import Flask
from flask import render_template,request
from transformers import pipeline


# In[2]:


app = Flask(__name__)


# In[3]:


classifier = pipeline('sentiment-analysis', "mrm8488/bert-small-finetuned-squadv2")


# In[4]:


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        input_txt = request.form.get("input_txt")
        print(input_txt)
        r = classifier(input_txt)
        return(render_template("index.html",result=r))
    else:
        return(render_template("index.html",result="Waiting"))


# In[5]:


if __name__=="__main__":
    app.run()


# In[ ]:




