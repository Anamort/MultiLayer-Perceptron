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

public class MultilayerPerceptron {
    public static final double LEARNING_RATE = 0.5;
    public static final double TRAINING_TIME = 3000;
    public static final int HIDDEN_LAYER_COUNT = 1;
    public static final int HIDDEN_LAYER_NODE_COUNT = 4;

    List<String> data;
    int featureCount; //for input nodes
    int classCount; //for output nodes
    List <String> classNames;
    Layer inputLayer;
    List <Layer> hiddenLayers;
    Layer outputLayer;
    List <Double> targetValues; //based on the output nodes
    List <Target> targets;
    PrintWriter writer;

    private class Target{
        double value;
        String name;
        public Target(String title){
            this.name = title;
        }

        public double getValue() {
            return value;
        }

        public void setValue(double value) {
            this.value = value;
        }
    }

    public MultilayerPerceptron() {
        classNames = new ArrayList<String>();
        int depth = 0;
        setInputAndOutputLayers();
        inputLayer = new Layer(Layer.LayerType.INPUT, depth);
        List<Node> nodes = nodeCreation(featureCount, inputLayer); //will be defined by dataset
        inputLayer.setLayerNodes(nodes);
        depth++;
        hiddenLayers = new ArrayList<Layer>();
        for (int i=0; i<HIDDEN_LAYER_COUNT;i++) {
            Layer hiddenLayer = new Layer(Layer.LayerType.HIDDEN, depth);
            List<Node> hiddenNodes = nodeCreation(HIDDEN_LAYER_NODE_COUNT, hiddenLayer);
            hiddenLayers.add(hiddenLayer);
            hiddenLayer.setLayerNodes(hiddenNodes);
            if (i > 0){
                hiddenLayer.setUpLayer(hiddenLayers.get(i-1));
            }
            else if (i == 0){
                hiddenLayer.setUpLayer(inputLayer);
            }
            depth++;
        }

        outputLayer = new Layer(Layer.LayerType.OUTPUT, depth);
        List<Node> outputNodes = nodeCreation(classCount, outputLayer); //will be defined by dataset
        targets = new ArrayList<Target>();
        outputLayer.setLayerNodes(outputNodes);
        for (int i=0; i<classCount;i++) {
            Node node = outputNodes.get(i);
            node.setName(classNames.get(i));
            Target target = new Target(classNames.get(i));
            targets.add(target);
        }
        outputLayer.setUpLayer(hiddenLayers.get(hiddenLayers.size()-1)); // get last element

        inputLayer.setSubLayer(hiddenLayers.get(0));

        for (int i=0; i<HIDDEN_LAYER_COUNT-1;i++){ //last element has a special case
            Layer hiddenLayer = hiddenLayers.get(i);
            hiddenLayer.setSubLayer(hiddenLayers.get(i+1));
        }
        hiddenLayers.get(hiddenLayers.size()-1).setSubLayer(outputLayer); //the last element of hiddenLayers


        //initiliaze target values based on the data
    }

    public void setInputAndOutputLayers(){
        //String filePath = "testData.txt";
        //String filePath = "normalizedTestData";
        String filePath = "xorData";
        List<String> headerInfo = new ArrayList<String>();
        try
        {
            writer = new PrintWriter("plotData.txt");
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
                classCount = Integer.parseInt(numberOfFeatures[1]);
            }
            else{
                for (int j=1;j<=classCount;j++){
                    classNames.add(numberOfFeatures[j]);
                }
            }
        }
    }

    public void learningPhase(){

        for (int i=0; i< TRAINING_TIME; i++){
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
            System.out.println("For i: "+i);

            double theError = 0.0;
            for (Node node: outputLayer.getLayerNodes()) {
                theError = theError + node.errorRate;
            }
            //averageError = averageError / outputLayer.getNumberOfNodes();
            //System.out.println("The ERROR: "+theError/classCount);
            System.out.println("******************************************************************** ");
            /*
            if (averageError < 0.01)
                break;
                */
        }

    }

    public void testPhase(){

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
        targetValues = new ArrayList<Double>();
        for (Target target:targets){
            if (className.equals(target.name)){
                target.setValue(1.0);
            }else{
                target.setValue(0.0);
            }
        }
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
        Double averageError = 0.0;
        for (int i=0; i< outputLayer.getNumberOfNodes(); i++) {
            Node node = outputLayer.getLayerNodes().get(i);
            //delta.add(i, (targets.get(i).getValue() - node.getOutputValue())*(1 - Math.pow(node.getOutputValue(),2)));
            delta.add(i, (targets.get(i).getValue() - node.getOutputValue())*node.getOutputValue()*(1-node.getOutputValue()));
            node.errorRate = delta.get(i);
            System.out.println("Target value: "+targets.get(i).getValue()+"| | Node value: "+node.getOutputValue());
            node.errorRate = targets.get(i).getValue() - node.getOutputValue();
            System.out.println("Error is: "+ node.errorRate);
            averageError = averageError + Math.abs(node.errorRate);
        }
        System.out.println("Average error: "+averageError.floatValue()/outputLayer.getNumberOfNodes());
        writer.println(averageError.floatValue()/outputLayer.getNumberOfNodes());
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
                    errorChangeForNode = errorChangeForNode + (oldWeight * (delta.get(counter)));
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
