
# importing the necessary dependencies

from sentence_transformers import SentenceTransformer, util
from flask import Flask, render_template, request
from flask_cors import cross_origin
import pickle
from elasticsearch import Elasticsearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth
import nltk
import numpy as np

embedder = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

app = Flask(__name__) # initializing a flask app

@app.route('/',methods=['GET'])  # route to display the home page
@cross_origin()
def homePage():
    return render_template("index.html")

@app.route('/predict',methods=['POST','GET']) # route to show the predictions in a web UI
@cross_origin()
def index():
    if request.method == 'POST':
        try:

            gre_score=request.form['query']
            
            query_embedding = embedder.encode(str(gre_score), convert_to_tensor=True)
            #cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
            #cos_scores = cos_scores.cpu()
            #top_results = np.argpartition(-cos_scores, range(top_k))[0:top_k]
            #b=[]
            #for idx in top_results[0:top_k]:
               # print(doc6[idx].strip(), "(Score: %.4f)" % (cos_scores[idx]))
               # a=doc6[idx].strip()
               # b.append(a)
            #r=es.search(index="searchbot", body={"query":{"match":{"attachment.content":{"query":b[0]}}},"_source": False})

            return render_template('results.html',prediction=gre_score)
        except Exception as e:
            print('The Exception message is: ',e)
            return 'something is wrong'
    # return render_template('results.html')
    else:
        return render_template('index.html')



if __name__ == "__main__":
    #app.run(host='127.0.0.1', port=8001, debug=True)
	app.run(debug=True) # running the app