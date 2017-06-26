import sys
import json
from jsonrpc import ServerProxy, JsonRpc20, TransportTcpIp
from pprint import pprint

class StanfordNLP:
    def __init__(self):
        self.server = ServerProxy(JsonRpc20(),
                                  TransportTcpIp(addr=("127.0.0.1", 8080)))
    
    def parse(self, text):
        return json.loads(self.server.parse(text))

nlp = StanfordNLP()
result = nlp.parse("Aaron renounces an enterprise against Jake, father of Sally, the girl he loves, when Jake withdraws his objections to Aaron and allows him to marry her.")
pprint(result)

from nltk.tree import Tree
tree = Tree.fromstring(result['sentences'][0]['parsetree'])
pprint(tree)
