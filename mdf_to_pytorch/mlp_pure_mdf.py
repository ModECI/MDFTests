from modeci_mdf.mdf import *
from modeci_mdf.simple_scheduler import EvaluableGraph
from modeci_mdf.export.graphviz import mdf_to_graphviz

import numpy as np
import sys
import h5py
import time

def get_weight_info():

    weights = {}
    f = h5py.File('example_mdfs/mlp_classifier/weights.h5', 'r')

    for key in list(f.keys()):
        weight_mat = f[key][:]
        weights[key] = weight_mat
        print('Loaded %s: %s'%(key, weight_mat.shape))
    f.close()

    return weights

def get_model_graph():

    mod = Model(id='mlp_mdf_classifier')
    mod_graph = Graph(id=mod.id)
    mod.graphs.append(mod_graph)

    weights = get_weight_info()

    dim0 = 16
    weight = weights['weights.mlp_classifier.graphs.mlp_classifier.nodes.mlp_input_layer.parameters.weight']
    bias = weights['weights.mlp_classifier.graphs.mlp_classifier.nodes.mlp_input_layer.parameters.bias']

    dummy_input = np.zeros((1,14*14))
    #dummy_input = np.ones((14*14))

    input_node = Node(
        id="mlp_input_layer",
        parameters={"input": dummy_input,
                    "weight":weight.T,
                    "bias":bias.T},
    )

    f1 = Function(id="mul", function="MatMul", args={"A": 'input', "B": "weight"} )

    input_node.functions.append(f1)

    f2 = Function(id="sum", function="linear",
        args={"variable0": 'mul', "slope": 1, "intercept": "bias"})
    input_node.functions.append(f2)

    input_node.output_ports.append(OutputPort(id="out_port", value='sum'))
    mod_graph.nodes.append(input_node)


    relu1_node = Node(id="mlp_relu_1")
    relu1_node.input_ports.append(InputPort(id="in_port"))
    mod_graph.nodes.append(relu1_node)

    f1 = Function(id="relu1", function="Relu", args={"A": "in_port"})
    relu1_node.functions.append(f1)

    relu1_node.output_ports.append(OutputPort(id="out_port", value=f1.id))

    e1 = Edge(id="edge_1",
        sender=input_node.id,
        sender_port=input_node.output_ports[0].id,
        receiver=relu1_node.id,
        receiver_port=relu1_node.input_ports[0].id,
    )
    mod_graph.edges.append(e1)


    weight = weights['weights.mlp_classifier.graphs.mlp_classifier.nodes.mlp_hidden_layer_with_relu.parameters.weight']
    bias = weights['weights.mlp_classifier.graphs.mlp_classifier.nodes.mlp_hidden_layer_with_relu.parameters.bias']

    hr_node = Node(id="mlp_hidden_layer_with_relu",
        parameters={"weight":weight.T,
                    "bias":bias.T},)
    mod_graph.nodes.append(hr_node)
    hr_node.input_ports.append(InputPort(id="in_port"))

    f1 = Function(id="mul", function="MatMul", args={"A": 'in_port', "B": "weight"} )
    hr_node.functions.append(f1)

    f2 = Function(id="sum", function="linear",
                  args={"variable0": 'mul', "slope": 1, "intercept": "bias"})
    hr_node.functions.append(f2)

    f3 = Function(id="relu2", function="Relu", args={"A": "sum"})
    hr_node.functions.append(f3)

    hr_node.output_ports.append(OutputPort(id="out_port", value="relu2"))

    e2 = Edge(id="edge_2",
        sender=relu1_node.id,
        sender_port=relu1_node.output_ports[0].id,
        receiver=hr_node.id,
        receiver_port=hr_node.input_ports[0].id,
    )
    mod_graph.edges.append(e2)

    weight = weights['weights.mlp_classifier.graphs.mlp_classifier.nodes.mlp_output_layer.parameters.weight']
    bias = weights['weights.mlp_classifier.graphs.mlp_classifier.nodes.mlp_output_layer.parameters.bias']

    out_node = Node(id="mlp_output_layer",
        parameters={"weight":weight.T,
                    "bias":bias.T},)
    mod_graph.nodes.append(out_node)
    out_node.input_ports.append(InputPort(id="in_port"))

    f1 = Function(id="mul", function="MatMul", args={"A": 'in_port', "B": "weight"} )
    out_node.functions.append(f1)

    f2 = Function(id="sum", function="linear",
                  args={"variable0": 'mul', "slope": 1, "intercept": "bias"})
    out_node.functions.append(f2)

    out_node.output_ports.append(OutputPort(id="out_port", value="sum"))

    e3 = Edge(id="edge_3",
        sender=hr_node.id,
        sender_port=hr_node.output_ports[0].id,
        receiver=out_node.id,
        receiver_port=out_node.input_ports[0].id,
    )
    mod_graph.edges.append(e3)

    return mod_graph

def main():

    test_all = '-test' in sys.argv

    mod_graph = get_model_graph()

    mdf_to_graphviz(mod_graph,view_on_render=not test_all, level=3)

    from neuromllite.utils import FORMAT_NUMPY, FORMAT_TENSORFLOW

    format = FORMAT_TENSORFLOW if "-tf" in sys.argv else FORMAT_NUMPY
    eg = EvaluableGraph(mod_graph, verbose=False)
    eg.evaluate(array_format=format)

    print('Finished evaluating graph using array format %s'%format)

    for n in ['mlp_input_layer','mlp_relu_1','mlp_hidden_layer_with_relu','mlp_output_layer']:
        out = eg.enodes[n].evaluable_outputs['out_port'].curr_value
        print('Final output value of node %s: %s, shape: %s'%(n, out, out.shape))

    if test_all:
        # Iterate on training data, feed forward and log accuracy
        imgs = np.load("example_data/imgs.npy")
        labels = np.load("example_data/labels.npy")

        import torch.nn
        matches = 0
        imgs_to_test = imgs[:300]

        start = time.time()
        for i in range(len(imgs_to_test)):
            ii = imgs[i,:,:]
            target = labels[i]
            img = torch.Tensor(ii).view(-1, 14*14).numpy()
            #plot_img(img, 'Post_%i (%s)'%(i, img.shape))
            print('***********\nTesting image %i (label: %s): %s\n%s'%(i,target,np.array2string(img,threshold=5, edgeitems=2),img.shape))
            #print(mod_graph.nodes[0].parameters['input'])
            mod_graph.nodes[0].parameters['input'] = img
            eg = EvaluableGraph(mod_graph, verbose=False)
            eg.evaluate(array_format=format)
            for n in ['mlp_output_layer']:
                out = eg.enodes[n].evaluable_outputs['out_port'].curr_value
                print('Output of evaluated graph: %s %s (%s)'%(out,out.shape,type(out).__name__))
                prediction = np.argmax(out)

            match = target==int(prediction)
            if match: matches+=1
            print('Target: %s, prediction: %s, match: %s'%(target, prediction, match))
        t = time.time()-start
        print('Matches: %i/%i, accuracy: %s%%. Total time: %.4f sec (%.4fs per run)'%(matches,len(imgs_to_test), (100.*matches)/len(imgs_to_test),t,t/len(imgs_to_test)))

if __name__ == "__main__":
    main()
