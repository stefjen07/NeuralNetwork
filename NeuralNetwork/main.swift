//
//  main.swift
//  NeuralNetwork
//
//  Created by Евгений on 15.04.2021.
//

import Foundation

//Test ( inputSize = 3, outputSize = 2 )

let usingFile = true
let eraseFile = false
let fileName = "testModel.nn"

var network = NeuralNetwork()

if usingFile && !eraseFile {
    network = NeuralNetwork(fileName: fileName)
} else {
    network.learningRate = 0.1
    network.epochs = 1000
    network.layers = [
        Dense(inputSize: 3, neuronsCount: 6, functionRaw: .sigmoid),
        Dropout(inputSize: 6, probability: 0.05),
        Dense(inputSize: 6, neuronsCount: 12, functionRaw: .sigmoid),
        Dense(inputSize: 12, neuronsCount: 2, functionRaw: .sigmoid)
    ]
}

let set = Dataset(items: [
    DataItem(input: [0.0, 0.0, 0.0], output: [0.0, 1.0]),
    DataItem(input: [0.0, 0.0, 1.0], output: [1.0, 0.0]),
    DataItem(input: [0.0, 1.0, 0.0], output: [0.0, 1.0]),
    DataItem(input: [1.0, 0.0, 0.0], output: [0.0, 1.0]),
    DataItem(input: [1.0, 1.0, 1.0], output: [1.0, 0.0])
])

network.train(set: set)

print("Prediction for [0,1,1]: ", network.predict(input: [0.0, 1.0, 1.0]))
print("Prediction for [1,0,1]: ", network.predict(input: [1.0, 0.0, 1.0]))
print("Prediction for [1,1,0]: ", network.predict(input: [1.0, 1.0, 0.0]))
print("Prediction for [1,1,1]: ", network.predict(input: [1.0, 1.0, 1.0]))

if usingFile {
    network.saveModel(fileName: fileName)
    print("Model saved.")
}
