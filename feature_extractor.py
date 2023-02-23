import os
import pickle

actors = os.listdir('data')

filenames = []

for actor in actors:
    for imgFile in os.listdir(os.path.join('data', actor)):
        filenames.append(os.path.join('data', actor, imgFile))

pickle.dump(filenames,open('filenames.pkl','wb'))
