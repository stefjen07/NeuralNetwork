//
//  main.swift
//  NeuralNetwork
//
//  Created by Евгений on 15.04.2021.
//

import Foundation
import NeuralNetworkLibrary

//Test ( inputSize = 3, outputSize = 2 )

let usingFile = true
let eraseFile = true
let fileName = "testModel.nn"

//cook()

var network = NeuralNetwork()

if usingFile && !eraseFile {
    network = NeuralNetwork(fileName: fileName)
} else {
    network.learningRate = 0.5
    network.epochs = 10
    network.layers = [
        Convolutional2D(filters: 1, kernelSize: 1, stride: 1, functionRaw: .reLU),
        Convolutional2D(filters: 1, kernelSize: 1, stride: 1, functionRaw: .reLU),
        Flatten(inputSize: 1024),
        Dropout(inputSize: 1024, probability: 0.1),
        Dense(inputSize: 1024, neuronsCount: 71, functionRaw: .reLU)
    ]
}

network.printSummary()

var set = getDS()

network.train(set: .init(items: .init(set.items.shuffled().dropFirst(16))))

if usingFile {
    network.saveModel(fileName: fileName)
    print("Model saved.")
}
