from cassandra.cluster import Cluster
from cassandra import ConsistencyLevel
from cassandra.query import SimpleStatement
from flask import Flask, request
from PIL import Image, ImageFilter
from redis import Redis, RedisError
import os
import sys
import socket
import time
import tensorflow as tf
import logging
log = logging.getLogger()
log.setLevel('INFO')
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
log.addHandler(handler)

# Connect to Redis
redis = Redis(host="redis", db=0, socket_connect_timeout=2, socket_timeout=2)

app = Flask(__name__)


### Model setup
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)
init = tf.global_variables_initializer()
saver = tf.train.Saver()
sess = tf.Session()
sess.run(init)
saver.restore(sess, "models/model1")




### Connect to the cassandra
KEYSPACE="mykeyspace_mnist"
def createKeySpace():
    cluster = Cluster(contact_points=['127.0.0.1'],port=9042)
    session = cluster.connect()

    log.info("Creating keyspace...")
    try:
        session.execute("""
            CREATE KEYSPACE %s
            WITH replication = { 'class': 'SimpleStrategy', 'replication_factor': '2' }
            """ % KEYSPACE)

        log.info("setting keyspace...")
        session.set_keyspace(KEYSPACE)

        log.info("creating table...")
        session.execute("""
            CREATE TABLE mytable (
                mykey text,
                col1 text,
                col2 int,
                PRIMARY KEY (mykey, col1)
            )
            """)
    except Exception as e:
        log.error("Unable to create keyspace")
        log.error(e)

# createKeySpace()

'''use python to delete the created table'''

def deleteTable():
    cluster = Cluster(contact_points=['127.0.0.1'],port=9042)
    session = cluster.connect()

    log.info("setting keyspace...")
    session.set_keyspace(KEYSPACE)

    try:
        log.info("Deleting a table...")
        session.execute('''DROP TABLE mytable''')
    except Exception as e:
        log.error("Unable to delete a table")
        log.error(e)

'''use python to delete the created keyspace'''

def deleteKeyspace():
    cluster = Cluster(contact_points=['127.0.0.1'],port=9042)
    session = cluster.connect()

    try:
        log.info("Deleting a keyspace...")
        session.execute('''DROP KEYSPACE %s''' % KEYSPACE)
    except Exception as e:
        log.error("Unable to delete a keyspace")
        log.error(e)



'''use python to insert a few records in our table'''
# insert the images' time, name and value 

def insertData(time, name, value):
    cluster = Cluster(contact_points=['127.0.0.1'],port=9042)
    session = cluster.connect()

    log.info("setting keyspace...")
    session.set_keyspace(KEYSPACE)

    prepared = session.prepare("""
    INSERT INTO mytable (mykey, col1, col2)
    VALUES (?, ?, ?)
    """)

    log.info("inserting into mytable")
    session.execute(prepared.bind((time, name, value)))
    # session.execute('''insert into mykeyspace.mytable(mykey,col1,col2) values(%s,%s,%d)''' %(time, name, value))

    # for i in range(number):
    #     if(i%5 == 0):
    #         log.info("inserting row %d" % i)
    #     session.execute(prepared.bind(("rec_key_%d" % i, 'aaa', 'bbb')))

# insertData("2018.4.17", "1.png", 1)
# insertData(20)


'''Reading the freshly inserted data is not that difficult using a function similar to the one below:'''
def readRows():
    cluster = Cluster(contact_points=['127.0.0.1'],port=9042)
    session = cluster.connect()

    log.info("setting keyspace...")
    session.set_keyspace(KEYSPACE)

    rows = session.execute("SELECT * FROM mytable")
    log.info("key\tcol1\tcol2")
    log.info("---------\t----\t----")

    count=0
    for row in rows:
        if(count%100==0):
            log.info('\t'.join(row))
        count=count+1

    log.info("Total")
    log.info("-----")
    log.info("rows %d" %(count))


###
@app.route("/prediction", methods=['GET','POST'])
def predictint():
    imname = request.files["file"]   
    file_name = request.files["file"].filename
    imvalu = prepareImage(imname)
    prediction = tf.argmax(y,1)
    pred = prediction.eval(feed_dict={x: [imvalu]}, session=sess)
    #get timestamp
    uploadtime=time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
    #store history
    session.execute("INSERT INTO mnist(id, digits, image_name, upload_time) values(uuid(), %s, %s, %s)",[int(str(pred[0])), file_name, uploadtime])
    return "The number upload is: [%s]" % str(pred[0])

def preprocessImage(i):
    im = Image.open(i).convert('L')
    width = float(im.size[0])
    height = float(im.size[1])
    newImage = Image.new('L', (28, 28), (255)) #creates white canvas of 28x28 pixels
    
    if width > height: #check which dimension is bigger
        #Width is bigger. Width becomes 20 pixels.
        nheight = int(round((20.0/width*height),0)) #resize height according to ratio width
        if (nheight == 0): #rare case but minimum is 1 pixel
            nheight = 1
        # resize and sharpen
        img = im.resize((20,nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wtop = int(round(((28 - nheight)/2),0)) #caculate horizontal pozition
        newImage.paste(img, (4, wtop)) #paste resized image on white canvas
    else:
        #Height is bigger. Height becomes 20 pixels. 
        nwidth = int(round((20.0/height*width),0)) #resize width according to ratio height
        if (nwidth == 0): #rare case but minimum is 1 pixel
            nwidth = 1
         # resize and sharpen
        img = im.resize((nwidth,20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wleft = int(round(((28 - nwidth)/2),0)) #caculate vertical pozition
        newImage.paste(img, (wleft, 4)) #paste resized image on white canvas
    
    tv = list(newImage.getdata()) #get pixel values
    
    #normalize pixels to 0 and 1. 0 is pure white, 1 is pure black.
    tva = [ (255-x)*1.0/255.0 for x in tv] 
    return tva  # Vector of values
@app.route('/')
def index():
    try:
        visits = redis.incr("counter")
    except RedisError:
        visits = "<i>cannot connect to Redis, counter disabled</i>"

    html = '''
    <!doctype html>
    <html>
    <body>
    <form action='/prediction' method='post' enctype='multipart/form-data'>
        <input type='file' name='file'>
    <input type='submit' value='Upload'>
    </form>
    '''   
    return html.format(name=os.getenv("NAME", "MNIST"), hostname=socket.gethostname(), visits=visits) 


if __name__ == "__main__":
   createKeySpace()
   # trainApp = TrainMnist(CKPT_DIR)
   # trainApp.train()
   # trainApp.calculate_accuracy()
   MNIST_app.run(host='0.0.0.0',port=8000,)
# expose to all, can be accessed by LAN users   
