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

var network = NeuralNetwork()

if usingFile && !eraseFile {
    network = NeuralNetwork(fileName: fileName)
} else {
    network.learningRate = 0.5
    network.epochs = 1000
    network.layers = [
        Convolutional2D(filters: 1, kernelSize: 1, stride: 1, functionRaw: .sigmoid),
        Flatten(),
        Dense(inputSize: 4, neuronsCount: 6, functionRaw: .sigmoid),
        Dropout(inputSize: 6, probability: 0.1),
        Dense(inputSize: 6, neuronsCount: 2, functionRaw: .sigmoid)
    ]
}

network.printSummary()

let set = Dataset(items: [
    DataItem(input: [0.0, 0.0, 0.0, 0.0], inputSize: .init(width: 2, height: 2), output: classifierOutput(classes: 2, correct: 1).body, outputSize: .init(width: 2)),
    DataItem(input: [0.0, 0.0, 0.0, 1.0], inputSize: .init(width: 2, height: 2), output: classifierOutput(classes: 2, correct: 0).body, outputSize: .init(width: 2)),
    DataItem(input: [0.0, 0.0, 1.0, 0.0], inputSize: .init(width: 2, height: 2), output: classifierOutput(classes: 2, correct: 1).body, outputSize: .init(width: 2)),
    DataItem(input: [0.0, 1.0, 0.0, 0.0], inputSize: .init(width: 2, height: 2), output: classifierOutput(classes: 2, correct: 1).body, outputSize: .init(width: 2)),
    DataItem(input: [1.0, 1.0, 1.0, 1.0], inputSize: .init(width: 2, height: 2), output: classifierOutput(classes: 2, correct: 0).body, outputSize: .init(width: 2))
])

network.train(set: set)

for x in 0...1 {
    for y in 0...1 {
        for z in 0...1 {
            for q in 0...1 {
                print("Prediction for [\(x),\(y),\(z),\(q)]: ", network.predict(input: .init(size: .init(width: 2, height: 2), body: [Float(x), Float(y), Float(z), Float(q)])))
            }
        }
    }
}

if usingFile {
    network.saveModel(fileName: fileName)
    print("Model saved.")
}

//Non-dropout
//1.7343979
//1.2855911

//Dropout
//1.3407227
//1.4222838
//1.6375583
