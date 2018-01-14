import cv2
import random
import simplejson as json
import eleonora.utils.config as config
from eleonora.utils.util import getJson, crop_center, getNames

class createDatabase(object):
    def __init__(self, path):
        self.path = path
        self.data = json.loads(getJson(path))

    def isEmpty(self, key):
        return False if len(self.data.get(key)) > 0 else True

    def read(self, key=False):
        if key == False:
            return self.data
        return self.data.get(key)

    def insert(self, key, data={}, saveImg=False, img=[], image_hash=None):
        try:
            self.data[key].append(data)
            self._saveToFile()

            if saveImg:
                faceFileName = "%s.jpg"%(image_hash.lower())
                cv2.imwrite('./eleonora/data/faces/' + faceFileName, img)

            return True, data, self.data
        except:
            return False, data, self.data


    def update(self):
        # TODO: Update db if needed
        print('update into db')

    def _saveToFile(self):
        with open(self.path, 'w', encoding='utf-8') as outfile:
            json.dump(self.data, outfile)


class Scheme(object):
    def __init__(self, scheme=None, data=[], image_hash=''):
        self.scheme = scheme
        self.data = data
        self.image_hash = image_hash

    def generateScheme(self):
        if self.scheme == 'person':
            scheme = self.generatePersonScheme()
        else:
            raise Exception('Incorrect scheme name')
        return scheme

    def generatePersonScheme(self):
        scheme = {
         "first_name": self.data[0],
         "last_name": self.data[1],
         "face": "%s.jpg"%(self.image_hash.lower())
        }
        return scheme


def insertPerson(data, db, sFile):
    if data == "":
        return 200, [], db
    first_name, last_name = data, ''
    image_hash = generateHash()

    personData = Scheme(scheme="person", data=(first_name, last_name), image_hash=image_hash).generateScheme()
    status, person, face_db = db.insert("persons", data=personData, saveImg=True, img=sFile, image_hash=image_hash)
    return status, person, face_db

def generateHash():
    return str(random.getrandbits(128))
