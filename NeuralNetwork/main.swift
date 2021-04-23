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
let eraseFile = false
let fileName = "testModel.nn"

//cook()

var network = NeuralNetwork()

if usingFile && !eraseFile {
    network = NeuralNetwork(fileName: fileName)
} else {
    network.learningRate = 0.5
    network.epochs = 10
    network.layers = [
        Convolutional2D(filters: 1, kernelSize: 1, stride: 1, functionRaw: .sigmoid),
        Pooling2D(kernelSize: 2, stride: 1, mode: .max, functionRaw: .sigmoid),
        Flatten(inputSize: 961),
        Dropout(inputSize: 961, probability: 10),
        Dense(inputSize: 961, neuronsCount: 46, functionRaw: .sigmoid)
    ]
}

network.printSummary()

var set = getDS()

let _ = network.train(set: set)

if usingFile {
    network.saveModel(fileName: fileName)
    print("Model saved.")
}
