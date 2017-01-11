from urlparse import urlparse, parse_qsl
from urllib import unquote_plus

class url_object(object):
    '''A url object that can be compared with other url orbjects
    without regard to the vagaries of encoding, escaping, and ordering
    of parameters in query strings.'''

    def __init__(self, url):
        parts = urlparse(url)
        _query = frozenset(parse_qsl(parts.query))
        _path = unquote_plus(parts.path)
        parts = parts._replace(query=_query, path=_path)
        self.parts = parts
        
    
    def __eq__(self, other):
        return self.parts == other.parts

    def distance_url(self, other):
        Me= self.parts
        You=other.parts
        S=0
        
        for i in range(6):
          S=S+float(Me[i]==You[i])
        return round(1-float(S/6),2)

          

    def __hash__(self):
        return hash(self.parts)


