import requests

url = 'http://localhost:8000/upload/'

path = '/Users/user/Downloads/хуйло.png'


files = {'files': open(path, 'rb')}

response = requests.post(url, files=files)
print(response.text)
