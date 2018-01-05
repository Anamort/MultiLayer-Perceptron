package com.company;

import java.io.*;
import java.lang.annotation.Target;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Stream;

public class SingleClassVersion {

        public static final double LEARNING_RATE = 0.5;
        public static final double TRAINING_TIME = 3000;
        public static final int HIDDEN_LAYER_COUNT = 1;
        public static final int HIDDEN_LAYER_NODE_COUNT = 4;

        List<String> data;
        int featureCount; //for input nodes
        final int classCount = 1; //for output nodes
        List <String> classNames;
        Layer inputLayer;
        List <Layer> hiddenLayers;
        Layer outputLayer;
        double targetValue; //based on the output nodes
        PrintWriter writer;
        double averageError;

    public SingleClassVersion() {
        classNames = new ArrayList<String>();
        int depth = 0;
        setInputAndOutputLayers();
        inputLayer = new Layer(Layer.LayerType.INPUT, depth);
        List<Node> nodes = nodeCreation(featureCount, inputLayer); //defined by dataset
        inputLayer.setLayerNodes(nodes);
        depth++;
        hiddenLayers = new ArrayList<Layer>();
        for (int i = 0; i < HIDDEN_LAYER_COUNT; i++) {
            Layer hiddenLayer = new Layer(Layer.LayerType.HIDDEN, depth);
            List<Node> hiddenNodes = nodeCreation(HIDDEN_LAYER_NODE_COUNT, hiddenLayer);
            hiddenLayers.add(hiddenLayer);
            hiddenLayer.setLayerNodes(hiddenNodes);
            if (i > 0) {
                hiddenLayer.setUpLayer(hiddenLayers.get(i - 1));
            } else if (i == 0) {
                hiddenLayer.setUpLayer(inputLayer);
            }
            depth++;
        }

        outputLayer = new Layer(Layer.LayerType.OUTPUT, depth);
        List<Node> outputNodes = nodeCreation(classCount, outputLayer); //defined by dataset
        outputLayer.setLayerNodes(outputNodes);
        for (int i = 0; i < classCount; i++) {
            Node node = outputNodes.get(i);
            node.setName(classNames.get(i));
        }
        outputLayer.setUpLayer(hiddenLayers.get(hiddenLayers.size() - 1)); // get last element

        inputLayer.setSubLayer(hiddenLayers.get(0));

        for (int i = 0; i < HIDDEN_LAYER_COUNT - 1; i++) { //last element has a special case
            Layer hiddenLayer = hiddenLayers.get(i);
            hiddenLayer.setSubLayer(hiddenLayers.get(i + 1));
        }
        hiddenLayers.get(hiddenLayers.size() - 1).setSubLayer(outputLayer); //the last element of hiddenLayers


        //initiliaze target values based on the data
    }

    public void setInputAndOutputLayers(){
        //String filePath = "testData.txt";
        //String filePath = "normalizedTestData";
        String filePath = "xorData";
        //String filePath = "andData";
        List<String> headerInfo = new ArrayList<String>();
        try
        {
            writer = new PrintWriter("Xor-plotData2ort-25-HL-1-HLC-4.txt");
        }
        catch (FileNotFoundException e)
        {
            e.printStackTrace();
        }
        data = new ArrayList<String>();
        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String currentLine;
            int counter = 0;
            while ((currentLine = br.readLine()) != null) {
                if (counter < 3){
                    headerInfo.add(currentLine);
                }else{
                    data.add(currentLine);
                }
                counter++;
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        for(int i=0;i<headerInfo.size();i++ ){
            String [] numberOfFeatures = headerInfo.get(i).split(",");
            if (i == 0){
                featureCount = Integer.parseInt(numberOfFeatures[1]);
            }
            else if(i==1){
                //classCount = Integer.parseInt(numberOfFeatures[1]); classCount must be 1 in this version
            }
            else{
                for (int j=1;j<=classCount;j++){
                    classNames.add(numberOfFeatures[j]);
                }
            }
        }
    }

    public void learningPhase(){
        averageError = 0.0;
        for (int i=0; i< TRAINING_TIME; i++){
            averageError = 0.0;
            for (String instance:data) {
                processInstance(instance);
                forwardPropagation();
                for (Node node: outputLayer.getLayerNodes()) {
                    //node.setOutputValue(Math.tanh(node.getOutputValue()));
                    node.setOutputValue(1 / (1 + Math.exp(-1*node.getOutputValue())));
                }
                backPropagation();
                inputLayer.resetValuesOfNodes();
                for (Layer hiddenLayer: hiddenLayers) {
                    hiddenLayer.resetValuesOfNodes();
                }

                //System.out.println("||||||||||||||||||| ");
                outputLayer.resetValuesOfNodes();
            }
            writer.println((averageError)/data.size());
            System.out.println("Dosyaya yazillan: "+(averageError)/data.size());

            System.out.println("For i: "+i);
            for (Node node: inputLayer.getLayerNodes()) {
                //System.out.println("Input Node: "+node+" weights are: "+node.weightList);
            }

            for (Layer hiddenLayer: hiddenLayers) {
                for (Node node: hiddenLayer.getLayerNodes()) {
                    //System.out.println("Hidden Node: "+node+" weights are: "+node.weightList);
                }
            }

            System.out.println("******************************************************************** ");
            /*
            if (averageError < 0.01)
                break;
                */
        }
        writer.close();

    }

    public void testPhase(){
        //out of scope for this project but it can be extended
    }

    public void processInstance(String instance){
        String [] features = instance.split(",");
        int index = 0;
        for (Node inputNode: inputLayer.getLayerNodes()) {
            Double feature = Double.parseDouble(features[index]);
            inputNode.setOutputValue(feature);
            index++;
        }
        String className = features[featureCount];
        Double feature =  Double.parseDouble(features[index]);
        targetValue = feature;
    }

    public void forwardPropagation(){
        List<Layer> layers = new ArrayList<Layer>();
        layers.add(inputLayer);
        for (Layer hiddenLayer: hiddenLayers) {
            layers.add(hiddenLayer);
        }
        //there is no need for outputLayer
        for (Layer layer: layers) {
            List<Node> nodes = layer.getLayerNodes();
            for (Node node: nodes) {
                //node.setOutputValue(Math.tanh(node.getOutputValue()));
                if (!layer.type.equals(Layer.LayerType.INPUT))
                node.setOutputValue(1 / (1 + Math.exp(-1*node.getOutputValue())));
                node.computeValues();
            }
        }

    }

    public void backPropagation(){
        List<Layer> layers = new ArrayList<Layer>();
        layers.add(inputLayer);
        for (Layer hiddenLayer: hiddenLayers) {
            layers.add(hiddenLayer);
        }
        List<Double> delta = new ArrayList<Double>();
        for (int i=0; i< outputLayer.getNumberOfNodes(); i++) {
            Node node = outputLayer.getLayerNodes().get(i);
            //delta.add(i, (targetValue - node.getOutputValue())*(1 - Math.pow(node.getOutputValue(),2)));
            delta.add(i, (targetValue - node.getOutputValue())*node.getOutputValue()*(1-node.getOutputValue()));
            System.out.println("Target value: "+targetValue+"| | Node value: "+node.getOutputValue());
            node.errorRate = targetValue - node.getOutputValue();
            System.out.println("Error is: "+ node.errorRate);
            averageError = averageError + Math.abs(node.errorRate);
        }

        for (int i=layers.size()-1;i >= 0; i--){
            Layer layer = layers.get(i);
            List<Node> nodeList = layer.getLayerNodes();
            List<Double> newDelta = new ArrayList<Double>();
            double errorChangeForNode = 0.0;
            int nodeCounter = 0;
            for (Node node: nodeList) {
                int counter = 0;
                for (Double deltaValue: delta) {
                    Double oldWeight = node.weightList.get(counter);
                    double changeOfWeight = LEARNING_RATE * delta.get(counter) * node.getOutputValue();
                    node.weightList.set(counter, (oldWeight + changeOfWeight));
                    errorChangeForNode = errorChangeForNode + (oldWeight * (delta.get(counter)*+1));
                    counter++;

                }
                //errorChangeForNode = errorChangeForNode * (1 - Math.pow(node.getOutputValue(),2));
                errorChangeForNode = errorChangeForNode * node.getOutputValue()*(1-node.getOutputValue());
                newDelta.add(nodeCounter, errorChangeForNode);
                node.errorRate = errorChangeForNode;
                nodeCounter++;
            }
            delta = new ArrayList<Double>(newDelta);
        }
    }

    public void startNeuralNetwork(){

        //***********First of all, weights are randomly selected************
        List<Node> inputNodes = inputLayer.getLayerNodes();
        for (Node node: inputNodes) {
            node.generateWeights();
            System.out.println("Input Node: "+node+" weights are: "+node.weightList);
        }
        for (Layer hiddenLayer: hiddenLayers) {
            List<Node> hiddenNodes = hiddenLayer.getLayerNodes();
            for (Node node: hiddenNodes) {
                node.generateWeights();
                System.out.println("Input Node: "+node+" weights are: "+node.weightList);
            }
        }
        List <Node> outputNodes = outputLayer.getLayerNodes();
        for (Node node: outputNodes) {
            node.generateWeights();
            System.out.println("Output Node: "+node+" name is: "+node.getName());
            System.out.println("Output Node: "+node+" weights are: "+node.weightList);
            System.out.println("Output Node: "+node+" value is: "+node.getOutputValue());
        }
        //***********First of all, weights are randomly selected************

        learningPhase();


    }

    public List<Node> nodeCreation(int nodeNumber, Layer layer){
        List<Node> nodes = new ArrayList<Node>(); //read from file?
        for (int i=0; i<nodeNumber;i++){
            Node node = new Node(layer);

            if(!layer.type.equals(Layer.LayerType.INPUT)){
                node.setOutputValue(0.0); //for initiliaziton
            }

            nodes.add(node);
        }
        return nodes;
    }

}
