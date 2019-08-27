import sys
from sklearn.metrics import roc_auc_score
import numpy as np
import random
import tensorflow as tf
'''
Load data
'''
input1=open(sys.argv[1])
input2=open(sys.argv[2])
sequence_length=int(sys.argv[3])
sequence_length2=int(sys.argv[4])
x=[]
y=[]
x2=[]
num=0
train_id=[]
tag_sequence=[]
for i in range(4):
    tag_sequence.append([])
for line in input1:
    line=line.strip().split(",")
    train_id.append(line[0])
    tmp=line[2].split()
    for i in range(len(tmp)):
        x.append(tmp[i])
    tmp=line[3].split()
    for i in range(len(tmp)):
        x2.append(tmp[i])
    y.append(float(line[1]))
    num=num+1
    tmp=line[4].split()
    for i in range(4):
        tag_sequence[i].append(tmp[i])
x_test=[]
y_test=[]
x2_test=[]
tag_sequence_test=[]
for i in range(4):
    tag_sequence_test.append([])
num2=0
test_id=[]
for line in input2:
    line=line.strip().split(",")
    test_id.append(line[0])
    tmp=line[2].split()
    for i in range(len(tmp)):
        x_test.append(tmp[i])
    tmp=line[3].split()
    for i in range(len(tmp)):
        x2_test.append(tmp[i])
    y_test.append(float(line[1]))
    num2=num2+1
    tmp=line[4].split()
    for i in range(4):
        tag_sequence_test[i].append(tmp[i])
y_test=np.array(y_test).reshape(num2,1)
x2_test=np.array(x2_test).reshape(num2,(len(line[3].split()))/sequence_length2,sequence_length2)
x2=np.array(x2).reshape(num,(len(line[3].split()))/sequence_length2,sequence_length2)
tag_sequence=np.array(tag_sequence).reshape(4,num,1)
tag_sequence_test=np.array(tag_sequence_test).reshape(4,num2,1)
y=np.array(y).reshape(num,1)
x=np.array(x).reshape(num,5,50,14)
x_test=np.array(x_test).reshape(num2,5,50,14)
'''
Parameter initializer
'''
batch_size = tf.placeholder(tf.int32,[])
input_time1=len(line[3].split())/sequence_length2
input_length1=sequence_length2
input_time2=len(line[2].split())/sequence_length
input_length2=sequence_length
rnn_units=128
lr=0.001
model_tag=tf.placeholder(tf.float32,[4,None,1])
model_x=tf.placeholder(tf.float32,[None,input_time1,input_length1])
model_y=tf.placeholder(tf.float32,[None,1])
model_multix=tf.placeholder(tf.float32,[None,5,input_time2/5,input_length2])
print input_time1,input_time2,input_length1,input_length2,sequence_length,sequence_length2
print x.shape,x2.shape,tag_sequence.shape
epoch=10
batch=128
'''
Time-aware LSTM cell
'''
class _Linear(object):

  def __init__(self,
               args,
               output_size,
               build_bias,
               bias_initializer=None,
               kernel_initializer=None):
    self._build_bias = build_bias

    if not nest.is_sequence(args):
      args = [args]
      self._is_sequence = False
    else:
      self._is_sequence = True

    # Calculate the total size of arguments on dimension 1.
    total_arg_size = 0
    shapes = [a.get_shape() for a in args]
    for shape in shapes:
      if shape.ndims != 2:
        raise ValueError("linear is expecting 2D arguments: %s" % shapes)
      if shape[1].value is None:
        raise ValueError("linear expects shape[1] to be provided for shape %s, "
                         "but saw %s" % (shape, shape[1]))
      else:
        total_arg_size += shape[1].value

    dtype = [a.dtype for a in args][0]

    scope = vs.get_variable_scope()
    with vs.variable_scope(scope) as outer_scope:
      self._weights = vs.get_variable(
          _WEIGHTS_VARIABLE_NAME, [total_arg_size, output_size],
          dtype=dtype,
          initializer=kernel_initializer)
      if build_bias:
        with vs.variable_scope(outer_scope) as inner_scope:
          inner_scope.set_partitioner(None)
          if bias_initializer is None:
            bias_initializer = init_ops.constant_initializer(0.0, dtype=dtype)
          self._biases = vs.get_variable(
              _BIAS_VARIABLE_NAME, [output_size],
              dtype=dtype,
              initializer=bias_initializer)

  def __call__(self, args):
    if not self._is_sequence:
      args = [args]

    if len(args) == 1:
      res = math_ops.matmul(args[0], self._weights)
    else:
      res = math_ops.matmul(array_ops.concat(args, 1), self._weights)
    if self._build_bias:
      res = nn_ops.bias_add(res, self._biases)
    return res

def _linear(args,
            output_size,
            bias,
            bias_initializer=None,
            kernel_initializer=None):
  total_arg_size = 0
  shapes = [a.get_shape() for a in args]
  for shape in shapes:
    if shape.ndims != 2:
      raise ValueError("linear is expecting 2D arguments: %s" % shapes)
    if shape[1].value is None:
      raise ValueError("linear expects shape[1] to be provided for shape %s, "
                       "but saw %s" % (shape, shape[1]))
    else:
      total_arg_size += shape[1].value

  dtype = [a.dtype for a in args][0]

  # Now the computation.
  scope = vs.get_variable_scope()
  with vs.variable_scope(scope) as outer_scope:
    weights = vs.get_variable(
        _WEIGHTS_VARIABLE_NAME, [total_arg_size, output_size],
        dtype=dtype,
        initializer=kernel_initializer)
    if len(args) == 1:
      res = math_ops.matmul(args[0], weights)
    else:
      res = math_ops.matmul(array_ops.concat(args, 1), weights)
    if not bias:
      return res
    with vs.variable_scope(outer_scope) as inner_scope:
      inner_scope.set_partitioner(None)
      if bias_initializer is None:
        bias_initializer = init_ops.constant_initializer(0.0, dtype=dtype)
      biases = vs.get_variable(
          _BIAS_VARIABLE_NAME, [output_size],
          dtype=dtype,
          initializer=bias_initializer)
    return nn_ops.bias_add(res, biases)
	
class LSTMStateTuple(_LSTMStateTuple):
  __slots__ = ()

  @property
  def dtype(self):
    (c, h) = self
    if c.dtype != h.dtype:
      raise TypeError("Inconsistent internal state: %s vs %s" %
                      (str(c.dtype), str(h.dtype)))
    return c.dtype

class timeawareLSTMCell(tf.nn.rnn_cell.RNNCell):
    def __init__(self, num_units, forget_bias=1.0,state_is_tuple=True, activation=None, reuse=None):
        super(timeawareLSTMCell, self).__init__(_reuse=reuse)
        self._num_units = num_units
        self._forget_bias = forget_bias
        self._state_is_tuple = state_is_tuple
        self._activation = activation or math_ops.tanh
        self._linear = None
        self._shortmemory_weight= tf.get_variable("weights_short", [num_units+1,num_units], initializer=tf.truncated_normal_initializer(stddev=0.1)) 
        self._longmemory_weight= tf.get_variable("weights_long", [num_units+1,num_units], initializer=tf.truncated_normal_initializer(stddev=0.1))
        self._shrinkmemory_weight= tf.get_variable("weights_shrink", [num_units+1,num_units], initializer=tf.truncated_normal_initializer(stddev=0.1))
        self._mergememory_weight= tf.get_variable("weights_merge", [num_units+1,num_units], initializer=tf.truncated_normal_initializer(stddev=0.1))
        self._shortmemory_bias=tf.Variable(tf.constant(0.1,shape=[num_units]), dtype=tf.float32)
        self._longmemory_bias=tf.Variable(tf.constant(0.1,shape=[num_units]), dtype=tf.float32)
        self._shrinkmemory_bias=tf.Variable(tf.constant(0.1,shape=[num_units]), dtype=tf.float32)
        self._mergememory_bias=tf.Variable(tf.constant(0.1,shape=[num_units]), dtype=tf.float32)
    def state_size(self):
        return (LSTMStateTuple(self._num_units, self._num_units))

    def output_size(self):
        return self._num_units

    def call(self, inputs, state):
        if self._state_is_tuple:
            c, h = state
        else:
            c, h = array_ops.split(value=state, num_or_size_splits=2, axis=1)
        #print tf.shape(inputs),inputs.shape
        inputs,inputs2=array_ops.split(inputs,[31,1],axis=1)
        if self._linear is None:
            self._linear = _Linear([inputs, h], 4 * self._num_units, True)
        i, j, f, o = array_ops.split(value=self._linear([inputs, h]), num_or_size_splits=4, axis=1)
        sigmoid = math_ops.sigmoid
        #print i.shape,j.shape
        shortmemory=sigmoid(math_ops.matmul(tf.concat([c,inputs2],1),self._shortmemory_weight)+self._shortmemory_bias)
        longmemory=sigmoid(math_ops.matmul(tf.concat([c,inputs2],1),self._longmemory_weight)+self._longmemory_bias)
        shrinkmemory=sigmoid(math_ops.matmul(tf.concat([shortmemory,inputs2],1),self._shrinkmemory_weight)+self._shrinkmemory_bias)
        c=sigmoid(math_ops.matmul(tf.concat([shrinkmemory,inputs2],1),self._mergememory_weight)+self._mergememory_bias)
        new_c=( c * sigmoid(f + self._forget_bias) + sigmoid(i) * self._activation(j))
        print c.shape,new_c.shape
        new_h = self._activation(new_c) * sigmoid(o)
        if self._state_is_tuple:
            new_state = LSTMStateTuple(new_c, new_h)
        else:
            new_state = array_ops.concat([new_c, new_h], 1)
        return new_h, new_state
'''
Model architecture
'''
def get_weight_variable(shape,regularizer=None):
    weights = tf.get_variable("weights", shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
    if regularizer != None:
        tf.add_to_collection('losses', regularizer(weights))
    return weights

def lstm_cell_origin(rnn_num):
    lstm_cell = timeawareLSTMCell(rnn_units, forget_bias=0.0, state_is_tuple=True)
    return lstm_cell

def lstm_model():
    global batch_size
    global batch_size,input_time1,input_length1,rnn_units,lr,model_x,model_y,epoch,model_tag
    lstm_cell1 = lstm_cell_origin(rnn_units)
    initial_state1=lstm_cell1.zero_state(batch_size,tf.float32)
    outputs1=[]
    state1=initial_state1
    lstm_cell2= lstm_cell_origin(rnn_units)
    initial_state2=lstm_cell2.zero_state(batch_size,tf.float32)
    outputs2=[]
    with tf.variable_scope("lstm1"):
        for time_step in range(input_time1):
            if time_step > 0: tf.get_variable_scope().reuse_variables()
            (cell_output1, state1) = lstm_cell1(model_x[:, time_step, :], state1) 
            outputs1.append(cell_output1)
    with tf.variable_scope("lstm2"):
        for i in range(5):
            if i>0:tf.get_variable_scope().reuse_variables()
            tmp_outputs, tmp_state = tf.nn.dynamic_rnn(lstm_cell2, inputs=model_multix[:,i,:,:], initial_state=initial_state2, time_major=False)
            outputs2.append(tmp_outputs[:,-1,:])  
    print "shape",outputs1[0].get_shape(),outputs2[1].get_shape
    print len(outputs1),len(outputs2)
    merged = tf.concat((outputs1[-1],outputs2[-1]),axis=1)
    lstm_cell3= lstm_cell_origin(rnn_units)
    state3=lstm_cell3.zero_state(batch_size,tf.float32)
    with tf.variable_scope("lstm3"):
        nn_lstm_weight=get_weight_variable([rnn_units,32])
        with tf.variable_scope("second_output_nn"):
            nn_lstm_weight2=get_weight_variable([32,1]) 
        #nn_lstm_bias=tf.Variable(tf.constant(0.1,shape=[32]), dtype=tf.float32)
        cl_result=[]
        for i in range(5):
            merged_input=tf.concat((outputs1[i],outputs2[i]),axis=1)
            if i==0:
                merged_input=tf.concat((merged_input,tf.fill([batch_size,1],-1.0)),axis=1)
            else:
                merged_input=tf.concat((merged_input,tf.fill([batch_size,1],-1.0)),axis=1)
            if i>0: 
                tf.get_variable_scope().reuse_variables()
            (cell_output3, state3) = lstm_cell3(merged_input, state3)
            pred_step=tf.nn.sigmoid(tf.matmul(cell_output3,nn_lstm_weight))
            with tf.variable_scope("second_output_nn"):
                pred_step=tf.nn.sigmoid(tf.matmul(pred_step,nn_lstm_weight2))
            if i<4:
                cl_result.append(pred_step)

    with tf.variable_scope("nn"):
        nn1_weights=get_weight_variable([rnn_units,32]) 
        nn1_bias=tf.Variable(tf.constant(0.1,shape=[32]), dtype=tf.float32)    
        pred=tf.nn.sigmoid(tf.matmul(cell_output3,nn1_weights)+nn1_bias)
    with tf.variable_scope("output_nn"):
        output_weights = get_weight_variable([32,1])
        output_bias=tf.Variable(tf.constant(0.1,shape=[1]), dtype=tf.float32)
        pred=tf.nn.sigmoid(tf.matmul(pred,output_weights)+output_bias)
    return pred,cl_result

'''
Model training
'''
def train_lstm():
    global batch_size,input_time1,input_length1,rnn_units,lr,model_x,model_y,epoch,model_tag
    global batch
    global x2
    global x2_test,y_test,y
    pred,cl_result=lstm_model()
    loss1=tf.losses.log_loss(model_y,pred)
    loss2=tf.losses.log_loss(model_tag[0],cl_result[0])
    for i in range(1,4):
        loss2=loss2+tf.losses.log_loss(model_tag[i],cl_result[i])
    output_prob=pred
    loss=loss1+loss2/10
    #correct_prediction = tf.equal(tf.round(output_prob), model_y)
    #accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))
    train_op = tf.train.RMSPropOptimizer(lr).minimize(loss) 
    #saver=tf.train.Saver(tf.global_variables())
    fetches = {
        "cost": [loss1,loss2],
        "prob": output_prob,
    }
    print x2.shape,y.shape,model_multix.shape
    print x2_test.shape,y_test.shape,model_multix.shape
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epo in range(epoch):
            start=0
            end=start+batch
            #print end,len(x2)
            while(end<len(x2)):
                #print epo,start
                train_info,loss1=sess.run([train_op,loss],feed_dict={model_x:x2[start:end],model_y:y[start:end],batch_size:batch,model_multix:x[start:end],model_tag:tag_sequence[:,start:end,:]})
                start=start+batch
                end=start+batch
            test_info=sess.run(fetches,feed_dict={model_x:x2_test,model_y:y_test,batch_size:len(y_test),model_multix:x_test,model_tag:tag_sequence_test})
            test_prop = map(lambda x: float(x), test_info['prob'].flatten().tolist())
            print epo,test_info['cost']
            print roc_auc_score(np.array(y_test).reshape(-1,),np.array(test_prop))
            
                    
train_lstm()                
