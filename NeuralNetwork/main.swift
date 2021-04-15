//
//  main.swift
//  NeuralNetwork
//
//  Created by Евгений on 15.04.2021.
//

import Foundation

//Test ( inputSize = 3, outputSize = 1 )

var network = NeuralNetwork()

network.learningRate = 0.1
network.epochs = 1000

network.layers = [
    Dense(neuronsCount: 6, inputSize: 3, function: Sigmoid()),
    Dense(neuronsCount: 12, inputSize: 6, function: Sigmoid()),
    Dense(neuronsCount: 2, inputSize: 12, function: Sigmoid())
]

let set = Dataset(items: [
    DataItem(input: [0.0, 0.0, 0.0], output: [0.0, 1.0]),
    DataItem(input: [0.0, 0.0, 1.0], output: [1.0, 0.0]),
    DataItem(input: [0.0, 1.0, 0.0], output: [0.0, 1.0]),
    DataItem(input: [1.0, 0.0, 0.0], output: [0.0, 1.0])
])

network.train(set: set)

print("Prediction: ", network.predict(input: [1.0, 0.0, 1.0]))
