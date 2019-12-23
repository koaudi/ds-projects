import requests

url = 'http://localhost:5000/results'
r = requests.post(url,json={'Male':5, 'Female':200, 'Unknown':400})

print(r.json())