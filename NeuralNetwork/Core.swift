//
//  Core.swift
//  NeuralNetwork
//
//  Created by Евгений on 15.04.2021.
//

import Foundation

func transferDerivative(output: Float) -> Float {
    return output * (1.0-output)
}

func outNeuron(_ neuron: Neuron, input: [Float]) -> Float {
    var out = neuron.bias
    for i in 0..<neuron.weights.count {
        out += neuron.weights[i] * input[i]
    }
    return out
}

struct DataItem {
    var input: [Float]
    var output: [Float]
}

struct Dataset {
    var items: [DataItem]
}

class NeuralNetwork {
    var layers: [Layer] = []
    var learningRate = Float(0.05)
    var epochs = 30
    
    func train(set: Dataset) {
        for epoch in 0..<epochs {
            var error = Float.zero
            for item in set.items {
                let predictions = forward(networkInput: item.input)
                for i in 0..<item.output.count {
                    error+=pow(item.output[i]-predictions[i], 2)
                }
                backward(expected: item.output)
                updateWeights(row: item.input)
            }
            print("Epoch \(epoch+1), error \(error).")
        }
    }
    
    func predict(input: [Float]) -> Int {
        let output = forward(networkInput: input)
        var maxi = 0
        for i in 1..<output.count {
            if(output[i]>output[maxi]) {
                maxi = i
            }
        }
        return maxi
    }
    
    func updateWeights(row: [Float]) {
        for i in 0..<layers.count {
            var input = row
            if i != 0 {
                input.removeAll()
                for neuron in layers[i-1].neurons {
                    input.append(neuron.output)
                }
            }
            for j in 0..<layers[i].neurons.count {
                for m in 0..<layers[i].neurons[j].weights.count {
                    layers[i].neurons[j].weights[m] += learningRate * layers[i].neurons[j].delta * input[m]
                }
                layers[i].neurons[j].bias += learningRate * layers[i].neurons[j].delta
            }
        }
    }
    
    func forward(networkInput: [Float]) -> [Float] {
        var input = networkInput, newInput = [Float]()
        for i in 0..<layers.count {
            newInput.removeAll()
            for j in 0..<layers[i].neurons.count {
                let output = outNeuron(layers[i].neurons[j], input: input)
                layers[i].neurons[j].output = layers[i].function.activation(input: output)
                newInput.append(layers[i].neurons[j].output)
            }
            input = newInput
        }
        return input
    }
    
    func backward(expected: [Float]) {
        for i in (0..<layers.count).reversed() {
            let layer = layers[i]
            var errors = [Float]()
            if i == layers.count-1 {
                for j in 0..<layer.neurons.count {
                    errors.append(expected[j]-layer.neurons[j].output)
                }
            } else {
                for j in 0..<layer.neurons.count {
                    var error = Float.zero
                    for neuron in layers[i+1].neurons {
                        error += neuron.weights[j]*neuron.delta
                    }
                    errors.append(error)
                }
            }
            for j in 0..<layer.neurons.count {
                layers[i].neurons[j].delta = errors[j] * transferDerivative(output: layers[i].neurons[j].output)
            }
        }
    }
}

struct Neuron {
    var weights: [Float]
    var bias: Float
    var delta: Float
    var output: Float
}

func derivative(output: Float) -> Float {
    return output*(1.0-output)
}
