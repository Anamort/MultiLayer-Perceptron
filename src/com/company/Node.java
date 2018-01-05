package com.company;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class Node {
    protected double inputValues;
    protected List<Double> weightList;
    protected int weightCount;
    protected Layer belongLayer;
    protected double outputValue;
    protected String name; //for output nodes indicating classes
    public double errorRate;

    public String getName() {
        if (belongLayer.type.equals(Layer.LayerType.OUTPUT)){
            return name;
        }
        return null;
    }

    public void setName(String name) {
        if (belongLayer.type.equals(Layer.LayerType.OUTPUT)){
            this.name = name;
        }
    }

    public double getOutputValue() {
        return outputValue;
    }

    public void setOutputValue(double outputValue) {
        this.outputValue = outputValue;
    }

    public Node(Layer layer){
        belongLayer = layer;

    }

    public int getWeightCount() {
        if (belongLayer.getSubLayer() != null){
            weightCount = belongLayer.getSubLayer().getNumberOfNodes();
        }
        return weightCount;
    }

    public void generateWeights(){
        if(belongLayer.getSubLayer() != null){
            weightList = new ArrayList<Double>();
            for(int i=0; i<getWeightCount(); i++){
                Random generator = new Random();
                double randomWeight = (generator.nextDouble()*2) - 1.0; //between [-1,1]
                //double randomWeight = (generator.nextDouble()*1); //between [0,1]
                weightList.add(randomWeight);
            }
        }

    }

    public void computeValues(){
        List<Node> subLayerNodes = belongLayer.getSubLayer().getLayerNodes();
        for(int i=0; i<getWeightCount(); i++){
            double value = subLayerNodes.get(i).getOutputValue();
            double weight = weightList.get(i);
            double newValue = value + (weight*outputValue);
            subLayerNodes.get(i).setOutputValue(newValue);
        }
    }


}
