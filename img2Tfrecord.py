"""
根据图片和图片标签生成TFRecord文件，以及根据TFRecord文件，解析出原始图片数据及标签信息，放入网络中训练
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import sys

# 图像和标签生成tfrecord
def image2tfrecord(image_list,label_list,filename):
    '''
    image_list:image path list
    label_list:label list
    '''
    length=len(image_list)
    writer=tf.python_io.TFRecordWriter(filename)
    for i in range(length):
        image=Image.open(image_list[i])
        if 'png' in image_list[i][-4:]:     # png格式的图片特殊，需要做判断
            if image.mode=='RGB':
                r, g, b = image.split()
                image = Image.merge("RGB", (r, g, b))
            elif image.mode=='L':   # 灰度图，可以正常加载
                pass
            else:       # 此时需要删除a通道
                r,g, b, a = image.split()
                image = Image.merge("RGB", (r, g, b))
        #这个地方就展开了
        image_bytes=image.tobytes()
        features={}
        features['image']=tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_bytes]))
        features['label']=tf.train.Feature(int64_list=tf.train.Int64List(value=[int(label_list[i])]))
        tf_features=tf.train.Features(feature=features)
        tf_example=tf.train.Example(features=tf_features)
        tf_serialized=tf_example.SerializeToString()
        writer.write(tf_serialized)
    writer.close()
    

#tfrecord解析为图片及标签
def pares_tf(example_proto):
    dics={}
    dics['label']=tf.FixedLenFeature((),dtype=tf.int64,default_value=0)
    dics['image']=tf.FixedLenFeature((),dtype=tf.string,default_value="")

    parsed_example=tf.parse_single_example(serialized=example_proto,features=dics)
    image=tf.decode_raw(parsed_example['image'],out_type=tf.uint8)
    #image=tf.image.decode_jpeg(parsed_example['image'], channels=1)
    #这个地方可以加一些操作
    
    image=tf.cast(image,tf.float32)/255
    image=tf.reshape(image,(28,28,1))
    image=pre_process(image)    # 图像增强
    #标签的操作
    label=parsed_example['label']
    label=tf.cast(label,tf.int32)
    label = tf.one_hot(label,depth=10,on_value=1.0,off_value=0.0)
    return image,label
    
    
# 图像预处理
def pre_process(images,random_flip_up_down=False,random_flip_left_right=False,random_brightness=True,random_contrast=True,random_saturation=False,random_hue=False):
    if random_flip_up_down:
        images = tf.image.random_flip_up_down(images)
    if random_flip_left_right:
        images = tf.image.random_flip_left_right(images)
    if random_brightness:
        images = tf.image.random_brightness(images, max_delta=0.2)
    if random_contrast:
        images = tf.image.random_contrast(images, 0.9, 1.1)
    if random_saturation:
        images = tf.image.random_saturation(images, 0.3, 0.5)
    if random_hue:
        images = tf.image.random_hue(images,0.2)
    new_size = tf.constant([28,28],dtype=tf.int32)
    images = tf.image.resize_images(images, new_size)
    return images
    
    
# tfrecord数据生成器
def dataset(tfrecordFile,batch_size,epochs):
    dataset=tf.data.TFRecordDataset(filenames=tfrecordFile)
    new_dataset=dataset.map(pares_tf)
    shuffle_dataset=new_dataset.shuffle(buffer_size=(100000))   # 多少个文件做shuffle
    batch_dataset=shuffle_dataset.batch(batch_size).repeat(epochs)
    batch_dataset=batch_dataset.prefetch(batch_size)
    iterator=batch_dataset.make_one_shot_iterator()
    next_element=iterator.get_next()
    return next_element
    

def main():
    tf.reset_default_graph()
    epochs = 15
    batch_size = 100
    total_sum = 0
    epoch = 0
    filenames=['train_{}.tfrecords'.format(i) for i in range(10)]
    next_element=dataset(filenames,batch_size=batch_size,epochs=epochs)

    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    train_num = mnist.train.num_examples

    input_data = tf.placeholder(tf.float32,shape=(None,28,28,1))
    input_label = tf.placeholder(tf.float32,shape=(None,10))

    hidden1 = tf.keras.layers.Conv2D(filters=16,kernel_size=3,strides=1,padding='valid',activation='relu')(input_data)
    hidden2 = tf.keras.layers.Conv2D(filters=32,kernel_size=3,strides=2,padding='same',activation='relu')(hidden1)
    hidden3 = tf.keras.layers.Conv2D(filters=64,kernel_size=3,strides=2,padding='valid',activation='relu')(hidden2)
    hidden4 = tf.keras.layers.MaxPool2D(pool_size=2)(hidden3)
    hidden5 = tf.keras.layers.Conv2D(filters=128,kernel_size=3,strides=2,padding='valid',activation='relu')(hidden4)
    hidden5 = tf.layers.Flatten()(hidden5)
    output = tf.keras.layers.Dense(10,activation='softmax')(hidden5)

    #损失函数
    loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(input_label,output))

    #优化器
    opt = tf.train.AdamOptimizer()
    train_op = opt.minimize(loss)

    # 测试评估
    acc = tf.reduce_mean(tf.keras.metrics.categorical_accuracy(input_label,output))
    #correct_pred = tf.equal(tf.argmax(input_label,axis=1),tf.argmax(output,axis=1))
    #acc = tf.reduce_mean(tf.cast(correct_pred,tf.float32))

    tf.add_to_collection('my_op',input_data)
    tf.add_to_collection('my_op',output)
    tf.add_to_collection('my_op',loss)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run([init])
        test_data = mnist.test.images
        test_label = mnist.test.labels
        test_data = test_data.reshape(-1,28,28,1)
        while epoch<epochs:
            data,label=sess.run([next_element[0],next_element[1]])
            total_sum+=batch_size
            sess.run([train_op],feed_dict={input_data:data,input_label:label})
            if total_sum//train_num>epoch:
                epoch = total_sum//train_num
                loss_val = sess.run([loss],feed_dict={input_data:data,input_label:label})
                acc_test = sess.run([acc],feed_dict={input_data:test_data,input_label:test_label})
                saver.save(sess, save_path="./model/my_model.ckpt")
                print('epoch:{},train_loss:{:.4f},test_acc:{:.4f}'.format(epoch,loss_val[0],acc_test[0]))
    

if __name__ == '__main__':
    main()