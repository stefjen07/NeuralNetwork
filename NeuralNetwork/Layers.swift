//
//  Layers.swift
//  NeuralNetwork
//
//  Created by Евгений on 15.04.2021.
//

import Foundation

protocol Layer {
    var neurons: [Neuron] { get set }
    var function: ActivationFunction { get set }
}

struct Dense: Layer {
    var neurons = [Neuron]()
    var function: ActivationFunction
    
    init(neuronsCount: Int, inputSize: Int, function: ActivationFunction) {
        for _ in 0..<neuronsCount {
            var weights = [Float]()
            for _ in 0..<inputSize {
                weights.append(Float.random(in: -1.0 ... 1.0))
            }
            neurons.append(Neuron(weights: weights, bias: 0.0, delta: 0.0, output: 0.0))
        }
        self.function = function
    }
}

protocol ActivationFunction {
    func activation(input: Float) -> Float
}

struct Sigmoid: ActivationFunction {
    func activation(input: Float) -> Float {
        return 1.0/(1.0+exp(-input))
    }
}
