from fastapi import FastAPI
from typing import List
from pydantic import BaseModel
from sentence_transformers import CrossEncoder
import numpy as np
from elasticsearch import Elasticsearch
import json

regions_config_json = open(
    "./regions_config.json")
regions_config = json.load(regions_config_json)
regions_config_json.close()

number_of_matches = 5

model = CrossEncoder('./TinyBERT-L-2_product_matcher_v4', max_length=60)

HOSTS = [
  # master nodes
  "http://10.0.0.129:9200",
  "http://10.0.0.130:9200",
  "http://10.0.0.131:9200",

  # coordianate node
  "http://10.0.0.125:9200",
  "http://10.0.0.126:9200",
  "http://10.0.0.128:9200",

  # data nodes
  "http://10.0.0.35:9200",
  "http://10.0.0.36:9200",
  "http://10.0.0.52:9200",
  "http://10.0.0.53:9200",
  "http://10.0.0.54:9200",
  "http://10.0.0.21:9200",
  "http://10.0.0.22:9200",
  "http://10.0.0.55:9200",
  "http://10.0.0.56:9200",
  "http://10.0.0.58:9200",
  "http://10.0.0.60:9200",
  "http://10.0.0.61:9200",
  "http://10.0.0.64:9200",
  "http://10.0.0.65:9200",
  "http://10.0.0.66:9200",
  "http://10.0.0.67:9200",
  "http://10.0.0.70:9200",
  "http://10.0.0.71:9200",
  "http://10.0.0.74:9200",
  "http://10.0.0.80:9200",
  "http://10.0.0.81:9200",
  "http://10.0.0.89:9200",
  "http://10.0.0.90:9200",
  "http://10.0.0.91:9200",
  "http://10.0.0.18:9200",
  "http://10.0.0.19:9200",
  "http://10.0.0.20:9200",
  "http://10.0.0.11:9200",
  "http://10.0.0.12:9200",
  "http://10.0.0.16:9200",
  "http://10.0.0.99:9200",
  "http://10.0.0.100:9200",
  "http://10.0.0.105:9200",
  "http://10.0.0.106:9200",
  "http://10.0.0.107:9200",
  "http://10.0.0.108:9200",
  "http://10.0.0.109:9200",
  "http://10.0.0.110:9200",
  "http://10.0.0.111:9200",
  "http://10.0.0.112:9200",

  "http://10.0.0.25:9200",
  "http://10.0.0.32:9200",
  "http://10.0.0.31:9200",
  "http://10.0.0.49:9200",
  "http://10.0.0.127:9200",
  "http://10.0.0.39:9200",
];

query_es = Elasticsearch(HOSTS, request_timeout=100000)

def product_name_query(website_id, name, websites):
    query = {
        "_source": [
            "products._meta.productName", "products._meta.images",
            "products.website", "eans", "products._meta.mpn", "id"
        ],
        "query": {
            "bool": {
                "must": [{
                    "match": {
                        "products._meta.productName": name
                    }
                }, {
                    "range": {
                        "eansLength": {
                            "gte": 1
                        }
                    }
                }
                ]
                ,"must_not": [
                    {"match": {"products.website": website_id}}
                ]
            }
        },
        "size":
            100,
        "min_score":
            10
    }

    if websites:
        if websites[0] != "all":
            query["query"]["bool"]["filter"] = [{
                "terms": {
                    "products.website":
                        [website["id"] for website in websites]
                }
            }]

    return query

def get_products_with_ean(website_id, name, websites,region):
    index = regions_config[region]["productmatches_region_index"]
    hits = query_es.search(body=product_name_query(website_id, name, websites), index=[index])["hits"]["hits"]
    probable_matches = {}

    for prod in hits:
        for p in prod["_source"]["products"]:
            probable_matches[p["_meta"]["productName"]] = prod["_source"]["eans"][0]
    
    return probable_matches


def result(product_no_ean, product_ean):
    matches = []
    
    lname_multiple = [product_no_ean] * len(product_ean)
    pairs = list(zip(lname_multiple, product_ean.keys()))
    scores = model.predict(pairs, batch_size=85)
    scores_indexes = np.argsort(scores)[::-1][:number_of_matches]
    
    for i in scores_indexes:
        matches.append({'product_name': pairs[i][1], 'ean': product_ean[pairs[i][1]], "score": str(scores[i])})

    return matches


class Data(BaseModel):
    product_name: str
    website_id: str
    websites: List[str]
    region: str

app = FastAPI()

@app.get("/")
async def home():
    return {"Check /matchingmodelapi and post a hashmap with a structure as follows : { 'product_name':String, 'website_id':String, 'websites':List[String], 'region':String }"}

@app.post("/matchingmodelapi")
async def synonyms(data: Data):
    import time
    start_time = time.time()
    print("Started for:", data.product_name)
    names = get_products_with_ean(data.website_id, data.product_name, data.websites, data.region)
    print(len(names))

    matches = result(data.product_name, names)

    print(f"It took {(time.time() - start_time)} seconds for : {data.product_name}")

    return ({'matches':matches, "time in seconds":time.time() - start_time})

print("Api is up and ready to work :)")