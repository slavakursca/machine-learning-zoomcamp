import requests

#local test
#url = 'http://localhost:9696/predict'
#docker test
url = 'http://0.0.0.0:9797/predict'

client = {
    "lead_source": "organic_search",
    "number_of_courses_viewed": 4,
    "annual_income": 80304.0
}
repsonse = requests.post(url, json=client).json()

print(repsonse)