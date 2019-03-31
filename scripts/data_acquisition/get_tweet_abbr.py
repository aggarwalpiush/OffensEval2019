# !usr/bin/env python
# *-- codng : utf-8 --*


from bs4 import BeautifulSoup
import requests, json


def main():
	resp = requests.get('http://www.netlingo.com/acronyms.php')
	soup = BeautifulSoup(resp.text, "html.parser")
	slangdict= {}
	key=""
	value=""
	for div in soup.findAll('div', attrs={'class':'list_box3'}):
		for li in div.findAll('li'):
		   for a in li.findAll('a'):
		       key =a.text
		   value = li.text.split(key)[1]
		   slangdict[key]=value
	with open('myslang.json', 'w') as fid:
		json.dump(slangdict,fid,indent=2)

if __name__ == '__main__':
	main()
