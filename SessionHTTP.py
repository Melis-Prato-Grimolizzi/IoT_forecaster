import requests

session = requests.Session()
daemonSession = requests.Session()
print("Session created!")

def getSession():
    return session