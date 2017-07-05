#!/usr/bin/env python
"""
Very simple HTTP server in python.
Usage::
    ./dummy-web-server.py [<port>]
Send a GET request::
    curl http://localhost
Send a HEAD request::
    curl -I http://localhost
Send a POST request::
    curl -d "foo=bar&bin=baz" http://localhost
"""
from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer
import SocketServer
import re

def convertHex(s):
    hexp = re.compile(r'\%([0-9]+)')
    match = hexp.search(s)
    while match is not None:
        start_full = match.start(0)
        start_num = match.start(1)
        end = match.end(0)
        num = match.group(1)
        char = num.decode('hex')
        s = s[0:start_full] + char + s[end:]
        match = hexp.search(s)
    return s

class MyHTTPRequestHandler(BaseHTTPRequestHandler):
    def _set_headers(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()

    def do_GET(self):
        self._set_headers()
        self.wfile.write("<html><body><h1>hi!</h1></body></html>")

    def do_HEAD(self):
        self._set_headers()
        
    def do_POST(self):
        print self.path;
        self._set_headers()
        self.wfile.write("<html><body><h1>POST!</h1></body></html>")
        filenameParser = re.compile(r'file\=([a-zA-Z0-9\-\_\%\.\/]+)')
        jsonParser = re.compile(r'program\=(\[\{[a-zA-Z0-9\%\:\,\/\{\}\[\]\-\_\.]+\}\])')
        filename_match = filenameParser.search(self.path)
        json_match = jsonParser.search(self.path)
        if filename_match is not None and json_match is not None:
            filename = filename_match.group(1)
            json = convertHex(json_match.group(1))
            if len(filename) > 0 and len(json) > 0:
                print "filename:", filename
                print "json:", json
                with open(filename, 'w') as outfile:
                    print >> outfile, json.strip()
        
server_address = ('', 8000)
httpd = HTTPServer(server_address, MyHTTPRequestHandler)
print 'Starting httpd...'
httpd.serve_forever()
