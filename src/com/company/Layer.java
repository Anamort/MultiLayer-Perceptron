package com.company;

import java.util.List;

public class Layer {
    enum LayerType{
        INPUT, HIDDEN, OUTPUT;
    }
    public LayerType type;
    public List<Node> layerNodes; //important for count
    public int layerDepth;
    public Layer subLayer;
    public Layer upLayer;

    public Layer getUpLayer() {
        return upLayer;
    }

    public void setUpLayer(Layer upLayer) {
        this.upLayer = upLayer;
    }

    public Layer(LayerType type, int depth) {
        this.type = type;
        layerDepth = depth;
    }

    public List<Node> getLayerNodes() {
        return layerNodes;
    }

    public void setLayerNodes(List<Node> layerNodes) {
        this.layerNodes = layerNodes;
    }

    public LayerType getType() {
        return type;
    }

    public int getLayerDepth() {
        return layerDepth;
    }

    public void setLayerDepth(int layerDepth) {
        this.layerDepth = layerDepth;
    }

    public Layer getSubLayer() {
        return subLayer;
    }

    public void setSubLayer(Layer subLayer) {
        this.subLayer = subLayer;
    }

    public int getNumberOfNodes(){
        return layerNodes.size();
    }

    public void resetValuesOfNodes(){
        for (Node node:layerNodes) {
            node.setOutputValue(0.0);
        }
    }
}
