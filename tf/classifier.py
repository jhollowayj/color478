import tensorflow as tf
import resnet


resnet_layers = 50

# only want to do this once
with open("../tensorflow-resnet/resnet-%d.tfmodel" % resnet_layers, mode='rb') as f:
    fileContent = f.read()
graph_def = tf.GraphDef()
graph_def.ParseFromString(fileContent)

tf.import_graph_def(graph_def, name="resnet")

class Classifier():

    def features(self, name, rgb):
        g = tf.get_default_graph()

        with tf.variable_scope(name):
            param_provider = CopyParamProvider()
            m = resnet.Model(param_provider)
            m.build(rgb, resnet_layers)

            scope = tf.get_variable_scope()

            def get_tensor(n):
                pool_name = '/'.join([scope.name, n])
                return g.get_operation_by_name(pool_name).outputs[0]

            pool1 = get_tensor("conv1/relu")
            pool2 = get_tensor("res2c/relu")
            pool3 = get_tensor("res3d/relu")
            pool4 = get_tensor("res4f/relu")
            pool5 = get_tensor("res5c/relu")

            self.feat = pool4

        return [ rgb, pool1, pool2, pool3, pool4, pool5 ] 
        

class CopyParamProvider():
    def __init__(self):
        self.context = tf.get_variable_scope().name

    def _get_tensor(self, name):
        scope = tf.get_variable_scope()
        # remove first part of scope name
        # e.g. self.context == "gen/gen_classifier"
        # and  scope.name == "gen/gen_classifier/conv1"
        # we want to remove the context part of the current scope
        assert scope.name.startswith(self.context)
        l = len(self.context)
        rest = scope.name[l:]

        tensor_name = 'resnet' + rest + '/' + name + ':0'
        return tf.get_default_graph().get_tensor_by_name(tensor_name)

    def mean_bgr(self):
        return self._get_tensor('mean_bgr')

    def conv_kernel(self, name, in_chans, out_chans, shape, strides):
        return self._get_tensor('kernel')

    def bn_params(self, bn_name, scale_name, d):
        mean = self._get_tensor('mean')
        var = self._get_tensor('var')
        scale = self._get_tensor('scale')
        offset = self._get_tensor('offset')
        return mean, var, scale, offset

    def fc_params(self, name):
        w = self._get_tensor('weights')
        b = self._get_tensor('bias')
        return w, b

