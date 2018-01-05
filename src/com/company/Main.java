package com.company;

import java.util.ArrayList;
import java.util.List;

public class Main {

    public static void main(String[] args) {

        /*
        1- It should be defined that if the set is a training or not
        2- Determination of the number of input nodes (features)
        3- Hidden Layers and their nodes
        4- Randomly assigned weight values
        5- Output nodes based on possible outcomes

        */

        //MultilayerPerceptron neuralNetwork = new MultilayerPerceptron(); //input, output degerleri file'a gore verilebilir
        //neuralNetwork.startNeuralNetwork();

        SingleClassVersion singleClass = new SingleClassVersion();
        singleClass.startNeuralNetwork();


    }



}
