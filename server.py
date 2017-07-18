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
import os


def convertHex(s):
    hexp = re.compile(r'\%([0-9][0-9])')
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
        cwd = os.getcwd()
        split_path = os.path.split(self.path)
        print split_path
        if len(split_path) > 1 and (split_path[1] == "mxClient.js" or 'images' in split_path[0]):
            file = open(cwd + self.path)
            out = file.read()
            file.close()
            self.wfile.write(out)
        else:
            print "Unauthorized access!"
            self.wfile.write("<html><body><h1>Don't do that!</h1></body></html>")
    

    def do_HEAD(self):
        self._set_headers()
        
    def do_POST(self):
        self._set_headers()
        saveFilenameParser = re.compile(r'save\=([a-zA-Z0-9\-\_\%\.\/]+)')
        loadFilenameParser = re.compile(r'load\=([a-zA-Z0-9\-\_\%\.\/]+)')
        jsonParser = re.compile(r'program\=(\[\{[\S]+\}\])')
        save_filename_match = saveFilenameParser.search(self.path)
        load_filename_match = loadFilenameParser.search(self.path)
        json_match = jsonParser.search(self.path)
        if save_filename_match is not None and json_match is not None:
            filename = save_filename_match.group(1)
            json = convertHex(json_match.group(1))
            if len(filename) > 0 and len(json) > 0:
                with open(filename, 'w') as outfile:
                    print >> outfile, json.strip()
            self.wfile.write("<html><body><h1>POST!</h1></body></html>")
        elif load_filename_match is not None:
            filename = load_filename_match.group(1)
            if os.path.exists(filename):
                file = open(filename, 'r')
                json = file.read()
                file.close()
                self.wfile.write(json)
            else:
                self.wfile.write("<html><body><h1>POST!</h1></body></html>")
        else:
            self.wfile.write("<html><body><h1>POST!</h1></body></html>")
        
programs_directory = './programs'
if not os.path.exists(programs_directory):
    os.makedirs(programs_directory)

server_address = ('', 8000)
httpd = HTTPServer(server_address, MyHTTPRequestHandler)
print 'Starting httpd...'
httpd.serve_forever()
