import json
import re
from xml.etree import ElementTree as etree
import os
import sys
from bs4 import BeautifulSoup

wikiDir = "" # The directory to find all the wiki files
outFilename = "" # The filename to dump plots to
titleFilename = "" # The filename to dump title names to

pattern = '*:plot'
breakSentences = True

okTags = ['b', 'i', 'a', 'strong', 'em']
listTags = ['ul', 'ol']
contentTypes = ['text', 'list']


# Check command line parameters
'''
if len(sys.argv) > 3:
	wikiDir = sys.argv[1]
	outFilename = sys.argv[2]
	titleFilename = sys.argv[3]
else:
	print "usage:", sys.argv[0], "wikidirectory resultfile titlefile"
	exit()
'''


########################
### HELPER FUNCTIONS

def matchPattern(pattern, str):
	if pattern == '*':
		return str
	else:
		pattern = pattern.split('|')
		for p in pattern:
			match = re.search(p+r'\b', str)
			if match is not None:
				return match.group(0)
	return ''


def fixList(soup, depth = 0):
	result = ''
	if soup.name is None:
		lines = [s.strip() for s in soup.splitlines()]
		for line in lines:
			if len(line) > 0 and line[0] == '-':
				dashes = ''
				for n in range(depth):
					dashes = dashes + '-'
				result = result + dashes + ' ' + line[1:].strip() + '\n'
	elif soup.name in listTags:
		for child in soup.children:
			result = result + fixList(child, depth + 1)
	return result



########################

def ReadWikipedia(wiki_directory, pattern, categoryPattern, out_file, titles_file, break_sentences = False):
	pattern = pattern.split(':')
	titlePattern = pattern.pop(0)
	headerList = pattern

	contentType = 'text' # The type of content to grab 'text' or 'list'. Should be the tail of the headerPattern.

	files = [] # All the wiki files

	# Get all the wiki files left by wikiextractor
	for dirname, dirnames, filenames in os.walk(os.path.join('.', wiki_directory)):
		for filename in filenames:
			if filename[0] != '.':
				files.append(os.path.join(dirname, filename))

	with open(out_file, "w") as outfile:
		# Opened the output file
		with open(titles_file, "w") as titlefile:
			# Opened the title file
			# Walk through each file. Each file has a json for each wikipedia article. Look for jsons with "plot" subheaders
			for file in files:
				#print >> outfile, "file:", file #FOR DEBUGGING
				data = [] # Each element is a json record
				# Read the file and get all the json records
				for line in open(file, 'r'):
					data.append(json.loads(line))
				# Look for pattern matches in heading tags inside the text of the json
				for j in data:
					# j is a json record
					titleMatch = matchPattern(titlePattern, j['title'])
					categoryMatch = matchPattern(categoryPattern, j['categories'])
					if len(titleMatch) > 0 and len(categoryMatch) > 0:
						print "title:", titleMatch, "in", j['title']
						print "category:", categoryMatch, "in", j['categories']
						# This json record is a match to titlePattern
						#print >> outfile, j['title'].encode('utf-8') #FOR DEBUGGING
						# Text element contains HTML
						soup = BeautifulSoup(j['text'].encode('utf-8'), "html.parser")
						result = "" # The result found (if any)
						inresult = False # Am I inside a result section of the article?
						previousHeaders = []
						headerIndex = 0
						# Walk through each element in the html soup object
						for n in range(len(soup.contents)):
							if inresult:
								print "in result"
							current = soup.contents[n] # The current html element
							if len(headerList) == 0:
								# If only title information is given, we just get everything
								if current is not None and current.name is None:
									result = result + current.strip() + ' '
							elif not inresult and current is not None and current.name is not None and current.name == 'h' + str(headerIndex + 2): # start with h2
								# Let's see if this header matches the current expected pattern
								#print >> outfile, "current(1):", current.name.encode('utf-8'), current.encode('utf-8')
								match = False
								if len(headerList) == 0:
									match = True
								elif len(headerList) > 0:
									match = matchPattern(headerList[headerIndex].lower(), current.get_text().lower())
								if match:
									# this header matches
									previousHeaders.append(current.get_text())
									#print >> outfile, "previousheaders(a):", map(lambda x: x.encode('utf-8'), previousHeaders)
									headerIndex = headerIndex + 1
									if headerIndex >= len(headerList):
										inresult = True
									elif headerList[headerIndex].lower() in contentTypes:
										inresult = True
										contentType = headerList[headerIndex].lower()
								else:
									previousHeaders = []
							elif inresult and current is not None and current.name is not None and current.name[0] == 'h' and int(current.name[1]) >= (headerIndex + 2):
								# I'm probably seeing a sub-heading inside of what I want
								previousHeaders.append(current.get_text())
								#print >> outfile, "previousheaders(b):", map(lambda x: x.encode('utf-8'), previousHeaders)
							elif inresult and current is not None and current.name is not None and current.name in listTags:
								# found a list inside what I am looking for
								result = result + '\n' + fixList(current) + '\n '
							elif inresult and current is not None and (current.name is None or current.name.lower() in okTags):
								# I'm probably looking at text inside of what I want
								#print >> outfile, "current(3):", current.encode('utf-8')
								if contentType != 'list':
									current = current.strip()
									# Sometimes we see the header name duplicated inside the text block that succeeds the sub-section header. Crop it off
									if len(current) > 0:
										if len(previousHeaders) > 0:
											#print >> outfile, "previousheaders(c):", map(lambda x: x.encode('utf-8'), previousHeaders)
											headerLength = reduce(lambda x,y: x+y, map(lambda z: len(z)+2, previousHeaders)) # add 2 for period and space.
											result = result + current[headerLength:].strip() + ' '
										else:
											result = result + current.strip() + ' '
										# Forget the previous header. It was either consumed or wasn't duplicated in the first place.
										previousHeaders = []
							elif inresult and current is not None and current.name is not None and current.name[0] == 'h' and int(current.name[1]) < (headerIndex + 2):
								# Probably left the block. All done with this json!
								print "leaving"
								break
							elif not inresult and current is not None and current.name is not None and current.name[0] == 'h' and int(current.name[1]) > 1 and int(current.name[1]) < (headerIndex + 2):
								# not in the result, but we went up one level
								headerIndex = headerIndex - 1
								if len(previousHeaders) > 0:
									previousHeaders.pop()
								# Let's see if this header matches the current expected pattern
								#print >> outfile, "current(2):", current.name.encode('utf-8'), current.encode('utf-8')
								match = matchPattern(headerList[headerIndex].lower(), current.get_text().lower())
								if len(match) > 0:
									# this header matches
									previousHeaders.append(current.get_text())
									#print >> outfile, "previousheaders(d):", map(lambda x: x.encode('utf-8'), previousHeaders)
									headerIndex = headerIndex + 1
									if headerIndex >= len(headerList):
										inresult = True
									elif headerList[headerIndex].lower() in contentTypes:
										inresult = True
										contentType = headerList[headerIndex].lower()
							elif not inresult and current is not None and current.name is not None and current.name[0] == 'h' and int(current.name[1]) > 1 and int(current.name[1]) >= (headerIndex + 2):
								# I'm not in the result block and I saw something that wasn't a header.
								previousHeaders = []
								#print >> outfile, "previous header cleared (e)", current.encode('utf-8')
							elif not inresult and current is not None and current.name is None and len(current.strip()) > 0:
								previousHeaders = []
								#print >> outfile, "previous header cleared (f)", current.encode('utf-8')
						# Did we find what we were looking for?
						if len(result) > 0:
							# ASSERT: I have a result
							# Record the name of the article with the result
							print >> titlefile, j['title'].encode('utf-8')
							# remove newlines
							#result = result.replace('\n', ' ').replace('\r', '').strip()
							# remove html tags (probably mainly hyperlinks)
							result = re.sub('<[^<]+?>', '', result)
							# remove character name initials and take periods off mr/mrs/ms/dr/etc.
							result = re.sub(' [M|m]r\.', ' mr', result)
							result = re.sub(' [M|m]rs\.', ' mrs', result)
							result = re.sub(' [M|m]s\.', ' ms', result)
							result = re.sub(' [D|d]r\.', ' dr', result)
							#result = re.sub(' [M|m]d\.', ' md', result)
							#result = re.sub(' [P|p][H|h][D|d]\.', ' phd', result)
							#result = re.sub(' [E|e][S|s][Q|q]\.', ' esq', result)
							result = re.sub(' [L|l][T|t]\.', ' lt', result)
							result = re.sub(' [G|g][O|o][V|v]\.', ' lt', result)
							result = re.sub(' [C|c][P|p][T|t]\.', ' cpt', result)
							result = re.sub(' [S|s][T|t]\.', ' st', result)
							# handle i.e. and cf.
							result = re.sub('i\.e\. ', 'ie ', result)
							result = re.sub('cf\. ', 'cf', result)
							# deal with periods in quotes
							result = re.sub('\.\"', '\".', result) 
							# remove single letter initials
							p4 = re.compile(r'([ \()])([A-Z|a-z])\.')
							result = p4.sub(r'\1\2', result) 
							# Acroymns with periods are not fun. Need two steps to get rid of those periods.
							# I don't think this is working quite right
							p1 = re.compile('([A-Z|a-z])\.([)|\"|\,])')
							result = p1.sub(r'\1\2', result)
							p2 = re.compile('\.([A-Z|a-z])')
							result = p2.sub(r'\1', result)
							# periods in numbers
							p3 = re.compile('([0-9]+)\.([0-9]+)')
							result = p3.sub(r'\1\2', result)
							# Print result
							if contentType == 'text':
								if break_sentences:
									parser = re.compile(r'([\?\.\!:])')
									lines = [s.strip() for s in parser.sub(r'\1\n', result).splitlines()]
									for line in lines:
										if len(line) > 0:
											print >> outfile, line.strip().encode('utf-8')
								else:
									print >> outfile, result.strip().encode('utf-8')
							elif contentType == 'list':
								lines = [s.strip() for s in result.splitlines()]
								for line in lines:
									if len(line) > 0 and line[0] == '-':
										print >> outfile, line.strip().encode('utf-8')
							print >> outfile, "<EOS>"


### TEST RUN
#readWikipedia(wikiDir, pattern, outFilename, titleFilename, breakSentences)




'''
TODO: future modules:
	- Remove Empty lines
	- Remove newlines
	- Segment sentences into separate lines
	- Remove characters/words/substrings
	- strip all lines
'''