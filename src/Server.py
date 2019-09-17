import json
import socket
import urllib
from http.server import BaseHTTPRequestHandler, HTTPServer, SimpleHTTPRequestHandler
import time
import calendar

import os
from socketserver import ThreadingMixIn

import errno
import qrcode
from Data.entities import worlds
from functools import reduce

hostName = "0.0.0.0"
hostPort = 8080
people_in_stores = {
"Castro": set([(1,1)]),
"FOX HOME": set([(1,1),(2,2),(3,3),(4,4)]),
"Avazi": set([(1,1),(2,2)])
}

category_to_time = {
    "clothing":5,
    "food":3,
    "drink":1,
    "coffee":1,
    "book":21,
    "wheels":45,
    "medicine":14,
    "locks":18
}


# generate qr
def getQR(id):
    print('getQR ({})'.format(id))
    img = qrcode.make(json.dumps({'id': id, 'creation_date': get_now()}))
    url = '{}.png'.format(id)
    img.save(url)
    print(url)
    return url


def get_now():
    return calendar.timegm(time.gmtime())


def scannedQR(store, id, creation_date):
    print('scannedQR ({}, {}, {})'.format(store, id, creation_date))
    categories = None
    for world in worlds.values():
        for s in world.get_services().values():
            if store == s.get_name():
                categories = s.categories
    l = [category_to_time[cat] for cat in categories]
    activity_time = reduce(lambda x, y: x + y, l) / len(l)
    # for (id,creation_date) in people_in_stores[store]:
    #     if creation_date + activity_time < get_now():
    #         people_in_stores[store].discard((id,creation_date))
    #         print("removed by time")
    if store not in people_in_stores:
        people_in_stores[store] = set()
    if (id,creation_date) in people_in_stores[store]:
        people_in_stores[store].discard((id,creation_date))
        print("removed")
        try:
            os.remove('{}.png'.format(id))
        except Exception as e:
            print(e)
    else:
        people_in_stores[store].add((id,creation_date))
        print("Added")

    # print(people_in_stores[store])
    return 'scanned successfully'


def clientLeft(params):
    pass


def getWorlds():
    print("getWorlds")
    return str([w.get_name() for w in worlds.values()])


def getWorld(id):
    print('getWorld ({})'.format(id))
    services = list(worlds[id].get_services().values())
    services = [s.__dict__() for s in services]
    for s in services:
        if s["name"] not in people_in_stores:
            people_in_stores[s["name"]] = set()
        s["crowd"] = len(people_in_stores[s["name"]])
        s["status"] = int(min(len(people_in_stores[s["name"]])/2,2))
    ser = [json.dumps(s) for s in services]
    return str(ser).replace("\'","")


def register(id):
    pass


def login(id):
    pass


functions = {
    "/getQR": getQR,
    "/scannedQR": scannedQR,
    "/clientLeft": clientLeft,
    "/getWorlds": getWorlds,
    "/getWorld": getWorld,
    "/register": register,
    "/login": login,
    "/": None
}


def parse_params(params):
    if params == '':
        return dict()
    params = [p.split('=') for p in params.split('&')]
    params = {k: v for k, v in params}
    return params


class MyServer(SimpleHTTPRequestHandler):
    def do_GET(self):
        try:
            url_parts = urllib.parse.unquote(self.path).split('?')
            req = url_parts[0]
            params_string = url_parts[1] if len(url_parts) >= 2 else ''
            if req not in functions.keys():
                print('{} not a supported function'.format(req))
                # if req[-10:] == 'index.html':
                #     print('copying...')
                #     with open(req) as f:
                #         self.copyfile(f, self.wfile)
                # else:
                return super().do_GET()
            params = parse_params(params_string)
            self.send_response(200)
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            print(params)
            self.wfile.write(functions[req](**params).encode("utf-8"))
        except TypeError as e:
            print(e)
            self.send_response(401)
            self.end_headers()
            self.wfile.write(e.__repr__().encode("utf-8"))
        except Exception as e:
            print(e)
            self.send_response(400)
            self.end_headers()
            self.wfile.write(e.__repr__().encode("utf-8"))
        return


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    pass


def get_server(port=8080, next_attempts=0, serve_path=None):
    Handler = MyServer
    if serve_path:
        Handler.serve_path = serve_path
    while next_attempts >= 0:
        try:
            httpd = ThreadingHTTPServer(("", port), Handler)
            return httpd
        except socket.error as e:
            if e.errno == errno.EADDRINUSE:
                next_attempts -= 1
                port += 1
            else:
                raise


def listen():
    print(time.asctime(), "Server Starts - %s:%s" % (hostName, hostPort))
    with get_server() as server:
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
    print(time.asctime(), "Server Stops - %s:%s" % (hostName, hostPort))


if __name__ == '__main__':
    listen()
